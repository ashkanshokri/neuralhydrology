
#!/bin/bash

cd /datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/workflows/camelaus_lstm

directory=configs/spatial_split/twofold

# Loop through YAML files and submit sbatch jobs
for CONFIG_FILE in $(find $directory -name "*.yml"); do
    echo $CONFIG_FILE
    sbatch --export=CONFIG_FILE=$CONFIG_FILE petrichore/job.slurm 
    
done
