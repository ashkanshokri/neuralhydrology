import re
from pathlib import Path
from typing import Dict

import torch

from neuralhydrology.training.train import start_training
from neuralhydrology.utils.config import Config


class PredictorNotFoundError(Exception):
    pass


def get_predictor_list(s: str) -> str:
    """
    Extracts predictor names from the input string and maps them to corresponding values.

    Parameters:
    - s (str): Input string containing predictor abbreviations.

    Returns:
    - str: Concatenated string of mapped predictor values.
    """
    predictor_names = re.findall(r'[A-Z][a-z]*', s)

    predictor_dict: Dict[str, str] = {
        'Pa': "precipitation_AWAP",
        'Em': "et_morton_point_SILO",
        # Add more mappings as needed
    }

    mapped_predictors = []

    for predictor in predictor_names:
        try:
            mapped_predictors.append(predictor_dict[predictor])
        except KeyError:
            raise PredictorNotFoundError(f"Predictor '{predictor}' not found in the dictionary.")

    return ''.join(mapped_predictors)


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
for predictors in ['P', 'PaEm', 'Em']:
    for fold in [0, 1]:
        _ = train_and_evaluate(fold, predictors, config_file)
