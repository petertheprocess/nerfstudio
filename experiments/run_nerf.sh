conda_setup
conda activate nerfstudio
experiment_name_prefix="ho3d_our_"
echo "=======SWITCHED TO NERFSTUDIO CONDA ENVIRONMENT======="
# Run the nerf cmd for each directory
count=0
# total number of directories
total=$(ls -l | grep -c ^d)
# only iterate over directories
for dir in */; do
    count=$((count+1))
    dir=${dir%*/} # remove trailing slash
    experiment_name="$experiment_name_prefix$dir"
    echo "=======Running NERF on $dir============="
    ns-train nerfacto --pipeline.model.predict-normals True \
    --experiment-name $experiment_name \
    --vis wandb --pipeline.model.background-color random \
    --pipeline.model.near-plane 0.15 --pipeline.model.far-plane 5 \
    --pipeline.model.camera-optimizer.mode off \
    nerfstudio-data --center-method none --auto-scale-poses False --orientation-method none --scene-scale 0.3 --data $dir.json
    echo "=======Finished running NERF on $dir, $count/$total========"
done
