# given 2 image, compute SSIM and PNSR over the region alpha in both images are not zero
from typing import Literal, Optional, Sequence, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torchmetrics.functional.image.ssim import _ssim_check_inputs, _ssim_compute
from torchmetrics.functional.image.utils import _gaussian_kernel_2d, _gaussian_kernel_3d, _reflection_pad_3d


def _mask_ssim_update(
    preds: Tensor,
    target: Tensor,
    mask: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    data_range: Optional[Union[float, Tuple[float, float]]] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    return_full_image: bool = False,
    return_contrast_sensitivity: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute Structural Similarity Index Measure.

    Args:
        preds: estimated image
        target: ground truth image
        mask: mask of the region to compute SSIM
        gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        data_range: Range of the image. If ``None``, it is determined from the image (max - min)
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the contrast term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``

    """
    is_3d = preds.ndim == 5
    mask = mask.float()
    if mask.shape != preds.shape:
        mask = mask.unsqueeze(1).expand_as(preds)

    if not isinstance(kernel_size, Sequence):
        kernel_size = 3 * [kernel_size] if is_3d else 2 * [kernel_size]
    if not isinstance(sigma, Sequence):
        sigma = 3 * [sigma] if is_3d else 2 * [sigma]

    if len(kernel_size) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(kernel_size) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )
    if len(sigma) != len(target.shape) - 2:
        raise ValueError(
            f"`kernel_size` has dimension {len(kernel_size)}, but expected to be two less that target dimensionality,"
            f" which is: {len(target.shape)}"
        )
    if len(sigma) not in (2, 3):
        raise ValueError(
            f"Expected `kernel_size` dimension to be 2 or 3. `kernel_size` dimensionality: {len(kernel_size)}"
        )

    if return_full_image and return_contrast_sensitivity:
        raise ValueError("Arguments `return_full_image` and `return_contrast_sensitivity` are mutually exclusive.")

    if any(x % 2 == 0 or x <= 0 for x in kernel_size):
        raise ValueError(f"Expected `kernel_size` to have odd positive number. Got {kernel_size}.")

    if any(y <= 0 for y in sigma):
        raise ValueError(f"Expected `sigma` to have positive number. Got {sigma}.")

    if data_range is None:
        data_range = max(preds.max() - preds.min(), target.max() - target.min())  # type: ignore[call-overload]
    elif isinstance(data_range, tuple):
        preds = torch.clamp(preds, min=data_range[0], max=data_range[1])
        target = torch.clamp(target, min=data_range[0], max=data_range[1])
        data_range = data_range[1] - data_range[0]

    c1 = pow(k1 * data_range, 2)  # type: ignore[operator]
    c2 = pow(k2 * data_range, 2)  # type: ignore[operator]
    device = preds.device

    channel = preds.size(1)
    dtype = preds.dtype
    gauss_kernel_size = [int(3.5 * s + 0.5) * 2 + 1 for s in sigma]

    pad_h = (gauss_kernel_size[0] - 1) // 2
    pad_w = (gauss_kernel_size[1] - 1) // 2

    if is_3d:
        pad_d = (gauss_kernel_size[2] - 1) // 2
        preds = _reflection_pad_3d(preds, pad_d, pad_w, pad_h)
        target = _reflection_pad_3d(target, pad_d, pad_w, pad_h)
        mask = _reflection_pad_3d(mask, pad_d, pad_w, pad_h)
        if gaussian_kernel:
            kernel = _gaussian_kernel_3d(channel, gauss_kernel_size, sigma, dtype, device)
    else:
        preds = F.pad(preds, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        target = F.pad(target, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        mask = F.pad(mask, (pad_w, pad_w, pad_h, pad_h), mode="reflect")
        if gaussian_kernel:
            kernel = _gaussian_kernel_2d(channel, gauss_kernel_size, sigma, dtype, device)

    if not gaussian_kernel:
        kernel = torch.ones((channel, 1, *kernel_size), dtype=dtype, device=device) / torch.prod(
            torch.tensor(kernel_size, dtype=dtype, device=device)
        )

    # mask processing
    preds_masked = preds * mask
    target_masked = target * mask
    # print(f"preds_masked: {preds_masked.shape}")
    # print(f"target_masked: {target_masked.shape}")
    # print(f"mask: {mask.shape}")

    input_list = torch.cat(
        (
            preds_masked,
            target_masked,
            preds_masked * preds_masked,
            target_masked * target_masked,
            preds_masked * target_masked,
        )
    )  # (5 * B, C, H, W)
    mask_float = mask.float()
    outputs = F.conv3d(input_list, kernel, groups=channel) if is_3d else F.conv2d(input_list, kernel, groups=channel)

    mask_avg = F.conv3d(mask_float, kernel, groups=channel) if is_3d else F.conv2d(mask, kernel, groups=channel)
    # repeat 5 times for each input
    mask_avg = mask_avg.repeat(5, 1, 1, 1)
    # Avoid division by zero
    mask_avg = torch.clamp(mask_avg, min=1.0)
    # Normalize the outputs with the mask
    outputs = outputs / mask_avg

    output_list = outputs.split(preds.shape[0])

    mu_pred_sq = output_list[0].pow(2)
    mu_target_sq = output_list[1].pow(2)
    mu_pred_target = output_list[0] * output_list[1]

    # Calculate the variance of the predicted and target images, should be non-negative
    sigma_pred_sq = torch.clamp(output_list[2] - mu_pred_sq, min=0.0)
    sigma_target_sq = torch.clamp(output_list[3] - mu_target_sq, min=0.0)
    sigma_pred_target = output_list[4] - mu_pred_target

    upper = 2 * sigma_pred_target.to(dtype) + c2
    lower = (sigma_pred_sq + sigma_target_sq).to(dtype) + c2

    ssim_idx_full_image = ((2 * mu_pred_target + c1) * upper) / ((mu_pred_sq + mu_target_sq + c1) * lower)
    # only consider the region where mask is not zero
    # print(ssim_idx_full_image.shape)
    # print(mask.shape)
    mask_nopad = mask[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d] if is_3d else mask[..., pad_h:-pad_h, pad_w:-pad_w]
    ssim_idx = ssim_idx_full_image * mask_nopad

    mask_factor = mask_nopad.reshape(mask_nopad.shape[0], -1).float().mean(-1)
    mask_factor = torch.where(mask_factor == 0, torch.ones_like(mask_factor), mask_factor)
    # print(f"mask_factor: {mask_factor}")

    if return_contrast_sensitivity:
        contrast_sensitivity = upper / lower
        if is_3d:
            contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w, pad_d:-pad_d]
        else:
            contrast_sensitivity = contrast_sensitivity[..., pad_h:-pad_h, pad_w:-pad_w]
        return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1) / mask_factor, contrast_sensitivity.reshape(
            contrast_sensitivity.shape[0], -1
        ).mean(-1)

    if return_full_image:
        return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1) / mask_factor, ssim_idx_full_image

    return ssim_idx.reshape(ssim_idx.shape[0], -1).mean(-1) / mask_factor


def mask_structural_similarity_index_measure(
    preds: Tensor,
    target: Tensor,
    mask: Tensor,
    gaussian_kernel: bool = True,
    sigma: Union[float, Sequence[float]] = 1.5,
    kernel_size: Union[int, Sequence[int]] = 11,
    reduction: Literal["elementwise_mean", "sum", "none", None] = "elementwise_mean",
    data_range: Optional[Union[float, Tuple[float, float]]] = None,
    k1: float = 0.01,
    k2: float = 0.03,
    return_full_image: bool = False,
    return_contrast_sensitivity: bool = False,
) -> Union[Tensor, Tuple[Tensor, Tensor]]:
    """Compute Structural Similarity Index Measure over the region where mask is not zero.

    Args:
        preds: estimated image
        target: ground truth image
        mask: mask of the region to compute SSIM
        gaussian_kernel: If true (default), a gaussian kernel is used, if false a uniform kernel is used
        sigma: Standard deviation of the gaussian kernel, anisotropic kernels are possible.
            Ignored if a uniform kernel is used
        kernel_size: the size of the uniform kernel, anisotropic kernels are possible.
            Ignored if a Gaussian kernel is used
        reduction: a method to reduce metric score over labels.

            - ``'elementwise_mean'``: takes the mean
            - ``'sum'``: takes the sum
            - ``'none'`` or ``None``: no reduction will be applied

        data_range:
            the range of the data. If None, it is determined from the data (max - min). If a tuple is provided then
            the range is calculated as the difference and input is clamped between the values.
        k1: Parameter of SSIM.
        k2: Parameter of SSIM.
        return_full_image: If true, the full ``ssim`` image is returned as a second argument.
            Mutually exclusive with ``return_contrast_sensitivity``
        return_contrast_sensitivity: If true, the constant term is returned as a second argument.
            The luminance term can be obtained with luminance=ssim/contrast
            Mutually exclusive with ``return_full_image``

    Return:
        Tensor with SSIM score

    Raises:
        TypeError:
            If ``preds`` and ``target`` don't have the same data type.
        ValueError:
            If ``preds`` and ``target`` don't have ``BxCxHxW shape``.
        ValueError:
            If the length of ``kernel_size`` or ``sigma`` is not ``2``.
        ValueError:
            If one of the elements of ``kernel_size`` is not an ``odd positive number``.
        ValueError:
            If one of the elements of ``sigma`` is not a ``positive number``.

    Example:
        >>> from torchmetrics.functional.image import structural_similarity_index_measure
        >>> preds = torch.rand([3, 3, 256, 256])
        >>> target = preds * 0.75
        >>> structural_similarity_index_measure(preds, target)
        tensor(0.9219)

    """
    preds, target = _ssim_check_inputs(preds, target)
    similarity_pack = _mask_ssim_update(
        preds,
        target,
        mask,
        gaussian_kernel,
        sigma,
        kernel_size,
        data_range,
        k1,
        k2,
        return_full_image,
        return_contrast_sensitivity,
    )

    if isinstance(similarity_pack, tuple):
        similarity, image = similarity_pack
        return _ssim_compute(similarity, reduction), image

    similarity = similarity_pack
    return _ssim_compute(similarity, reduction)


def mask_peak_signal_to_noise_ratio(
    target: Tensor, preds: Tensor, mask: Tensor, data_range: Optional[float] = None
) -> Tensor:
    """
    Computes the PSNR (Peak Signal-to-Noise Ratio) within the masked region of the images.

    Args:
        target (Tensor): The ground truth image with shape (N, C, H, W).
        preds (Tensor): The predicted image with shape (N, C, H, W).
        mask (Tensor): The binary mask indicating the region to compute PSNR over, with shape (N, 1, H, W).
        data_range (Optional[float]): The value range of the input images. If None, it defaults to the maximum value in the target tensor.

    Returns:
        Tensor: The computed PSNR value for each image in the batch, with shape (N,).
    """
    # Ensure that the target, preds, and mask tensors have the same shape
    # assert target.shape == preds.shape == mask.shape, "target, preds, and mask must have the same shape"
    assert target.shape == preds.shape, "target and preds must have the same shape"

    # If data_range is not provided, set it to the dynamic range of the target image
    if data_range is None:
        data_range = target.max() - target.min()

    # Compute the squared difference between the target and predicted images
    mse = (target - preds) ** 2

    # Apply the mask, so that only the masked region contributes to the MSE
    mse_masked = mse * mask

    # Compute the mean squared error within the masked region
    # mse_mean = mse_masked.sum(dim=[1, 2, 3]) / mask.sum(dim=[1, 2, 3])
    mse_mean = mse_masked.sum() / mask.sum()

    # Avoid division by zero by returning infinity when MSE is zero (perfect match)
    psnr = 10 * torch.log10((data_range**2) / mse_mean)

    return psnr.mean()


def ssim_mask(target, preds):
    if target.shape[1] == 4:
        mask1 = target[:, 3, :, :] > 0.2
        mask2 = preds[:, 3, :, :] > 0.2
        union_mask = mask1 + mask2
        # bool to float
        mask = union_mask.float()
    else:
        # mask is where the target is not zero
        mask1 = target.sum(dim=1) > 0
        mask2 = preds.sum(dim=1) > 0
        union_mask = mask1 + mask2
        # bool to float
        mask = union_mask.float()
    # mask = target[:, 3, :, :]  # get the alpha channel
    mask = mask.unsqueeze(1)
    mask = mask.expand_as(preds)
    return mask_structural_similarity_index_measure(target, preds, mask)


def psnr_mask(target, preds):
    """
    take gt's alpha channel as mask, compute psnr over the region where alpha is not zero
    then take gt's alpha channel and preds' alpha channel, compute psnr over alpha channel
    """
    # B C H W
    if target.shape[1] == 4:
        mask1 = target[:, 3, :, :] > 0.2
        # mask2 = preds[:, 3, :, :] > 0.2
        # union_mask = mask1 + mask2
        # bool to float
        # mask = union_mask.float()
        mask = mask1.float()
    else:
        # mask is where the target is not zero
        mask1 = target.sum(dim=1) > 0
        # bool to float
        mask = mask1.float()
    mask = mask.unsqueeze(1)
    mask = mask.expand_as(preds)
    rgb_psnr = mask_peak_signal_to_noise_ratio(target, preds, mask)
    return rgb_psnr


if __name__ == "__main__":
    preds = torch.rand([3, 4, 256, 256])
    mask = torch.zeros([4, 256, 256])
    mask[:, 128:, :] = 1
    mask = mask.unsqueeze(0)
    mask = mask.expand_as(preds)

    target = preds * mask
    # plot mask
    nomask = torch.ones_like(mask)

    print(mask_structural_similarity_index_measure(preds, target, mask))
    print(ssim_mask(target, preds))

    print(mask_peak_signal_to_noise_ratio(target, preds, nomask))
