conda_setup
conda activate nerfstudio
experiment_name_prefix="ho3d_our_2_"
outputs_dir="outputs"
echo "=======SWITCHED TO NERFSTUDIO CONDA ENVIRONMENT======="
# Run the nerf cmd for each directory
count=0
# total number of directories
total=$(ls -l | grep -c ^d)
# only iterate over directories except outputs
for experiment_dir in "$outputs_dir"/*; do
    if [[ "$base_experiment_dir" == "$experiment_name_prefix"* ]]; then
        nerfacto_path="$experiment_dir/nerfacto"
        # 检查是否存在 nerfacto 目录
        if [ -d "$nerfacto_path" ]; then
            # 遍历 nerfacto 文件夹中的所有时间戳子目录
            for time_dir in "$nerfacto_path"/*; do
                if [ -d "$time_dir" ]; then
                    # 找到 config.yml 文件
                    config_path="$time_dir/config.yml"
                    if [ -f "$config_path" ]; then
                        echo "Evaluating: $config_path"
                        ns-eval --load-config $config_path --output-path ./nerf_eval_report/$experiment_dir.json
                    fi
                fi
            done
        fi
    fi
done