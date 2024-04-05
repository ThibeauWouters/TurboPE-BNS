#!/bin/bash -l
#Set job requirements
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -p gpu
#SBATCH -t 60:00
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-gpu=1
#SBATCH --mem-per-gpu=15G

now=$(date)
echo "$now"

# Define dirs
export events_dir=$HOME/TurboPE-BNS/real_events
export this_dir=$events_dir/GW190425_NRTidalv2

# Loading modules
module load 2022
module load Python/3.10.4-GCCcore-11.3.0
conda activate /home/twouters2/miniconda3/envs/jim
 
# Copy necessary files
cp $this_dir/GW190425_snellius.py "$TMPDIR"
cp $events_dir/utils_plotting.py "$TMPDIR"

# Run the script
python $this_dir/GW190425_snellius.py

export final_output_dir="$this_dir/outdir_snellius/"
echo "Copying to: $final_output_dir"

#Copy output directory from scratch to home, but first check if exists
if [ -d "$final_output_dir" ]; then
    echo "Directory already exists: $final_output_dir"
else
    mkdir "$final_output_dir"
    echo "Directory created: $final_output_dir"
fi

echo "First running ls:"
ls $TMPDIR

echo "OK, copying now!"
cp -r $TMPDIR/* $final_output_dir

echo "DONE"