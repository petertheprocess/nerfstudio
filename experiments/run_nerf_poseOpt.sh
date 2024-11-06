conda_setup
conda activate nerfstudio
experiment_name_prefix="ho3d_our_poseOpt_"
echo "=======SWITCHED TO NERFSTUDIO CONDA ENVIRONMENT======="
# Run the nerf cmd for each directory
count=0
# total number of directories
total=$(ls -l | grep -c ^d)
echo "======TOTAL_Folder:$total"
# only iterate over directories except outputs
for dir in */; do
    if [ "$dir" == "outputs/" ]; then
        total=$((total-1))
        continue
    fi
    count=$((count+1))
    dir=${dir%*/} # remove trailing slash
    experiment_name="$experiment_name_prefix$dir"
    data_json="$dir/$dir.json"
    echo "=======Running NERF on $dir============="
    echo "======Data JSON: $data_json============"
    ns-train nerfacto --pipeline.model.predict-normals True \
    --experiment-name $experiment_name \
    --vis wandb --pipeline.model.background-color random \
    --pipeline.model.near-plane 0.10 --pipeline.model.far-plane 5 \
    --pipeline.model.camera-optimizer.mode SE3 \
    nerfstudio-data --center-method none --auto-scale-poses False --orientation-method none --scene-scale 1.0 --data $dir/$dir.json
    echo "=======Finished running NERF on $dir, $count/$total========"
done
