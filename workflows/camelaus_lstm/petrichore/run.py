from pathlib import Path
import torch

from neuralhydrology.nh_run import eval_run
from pathlib import Path
import torch
from neuralhydrology.utils.file_system_operation import get_latest_touched_directory
from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config


def get_predictor_list(s):
    perdictor_dict = {
        'p': ["precipitation_AWAP"],
        'e': ["et_morton_point_SILO"],
        'pe': ["precipitation_AWAP", "et_morton_point_SILO"]
    }
    return perdictor_dict[s]


def train_and_evaluate(fold, predictors, config_file):
    # Set up configuration
    cfg = Config(config_file)
    cfg.update_config({
        'dynamic_inputs': get_predictor_list(predictors),
        'train_basin_file': f'basins/twofoldsplit/fold_{fold}_train.txt',
        'test_basin_file': f'basins/twofoldsplit/fold_{fold}_test.txt',
        'validation_basin_file': f'basins/twofoldsplit/fold_{fold}_test.txt',
        'experiment_name': f'spatial_twofold_{fold}_{predictors}',
    })

    # Set device
    cfg.device = "cuda:0" if torch.cuda.is_available() else "cpu"

    # Start training
    start_training(cfg)



config_file = Path('configs/config.yml')
for predictors in ['p', 'pe', 'e']:
    for fold in [0, 1]:
        _ = train_and_evaluate(fold, predictors, config_file)
