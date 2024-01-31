#!/bin/bash

cd /datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/workflows/camelaus_lstm/petrichore
WD=/datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/workflows/camelaus_lstm/petrichore

directory=configs/spatial_split/twofold

# Loop through YAML files and submit sbatch jobs
for CONFIG_FILE in $(find $directory -name "*.yml"); do
    echo $CONFIG_FILE
    sbatch -D $WD --export=CONFIG_FILE=$CONFIG_FILE job.slurm 
    
done
