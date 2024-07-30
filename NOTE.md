data/pixel_samplers.py have a ignore_mask flag to ignore some pixels

in xxx_dataparser.py
eval_mode: Literal["fraction", "filename", "interval", "all"] = "fraction"
    """
    The method to use for splitting the dataset into train and eval.
    Fraction splits based on a percentage for train and the remaining for eval.
    Filename splits based on filenames containing train/eval.
    Interval uses every nth frame for eval.
    All uses all the images for any split.

## get_train_loss_dict in Pipeline

get ray_bundle from datamanager
```python
ray_bundle, batch = self.datamanager.next_train(step)
```
pixel_sampler is responsible for getting a batch of rays
2 parameters: ignore_mask and rejection_sample_mask
```python
    ignore_mask: bool = False
    """Whether to ignore the masks when sampling."""
    rejection_sample_mask: bool = True
    """Whether or not to use rejection sampling when sampling images with masks"""
```
calculate the model output
`model_output = self._model(ray_bundle)`

then the proposal_sampler is responsible for getting points along the rays, which yeilds ray_samples from ray_bundle
in Model base calss, base_model.py
```python
if self.collider is not None:
    ray_bundle = self.collider(ray_bundle)

return self.get_outputs(ray_bundle)
```

The collider is responsible for checking if the ray intersects with the scene
In nerfacto method, the collider is set default to NearFarCollider.
And the default
near_plane = 0.05
far_plane = 1000.0
只是给ray_bundle添加了一个near和far的属性，目前没有做实际的碰撞检测

ProposalSampler is responsible for sampling points along the rays
先经过SpacedSampler, 里面使用了near和far的属性，然后再经过PdfSampler

目前状态：如果把initial_samples设置为uniform，那么loss乱跳，全黑。


## 加入depth guide的sample
dataManager.train_dataset.cameras里可以得到相机的z轴，
对于每个ray，计算夹角算出对应的ray distance，然后在上面采样