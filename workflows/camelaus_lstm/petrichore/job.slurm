#!/bin/bash
#SBATCH --job-name=camelaus_lstm
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --hint=compute_bound
#SBATCH --mem=20000
#SBATCH --account=OD-221168
#SBATCH --time=24:00:00
#SBATCH --output=./reports/output_%a_%j.o
#SBATCH --error=./reports/error_%a_%j.e



source /datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/.venv/bin/activate

cd /datasets/work/d61-coastal-forecasting-wp3/work/sho108/neuralhydrology/workflows/camelaus_lstm/petrichore

data_dir=/datasets/work/lw-hydrofct/work/template/Data/CAMELS_AUS
echo $CONFIG_FILE
python -m neuralhydrology.nh_run_data_dir train --config-file $CONFIG_FILE --data-dir $data_dir
