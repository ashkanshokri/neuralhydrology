import os
import pickle
import re
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

import pandas as pd

# Local application/library specific imports
from neuralhydrology.nh_run_data_dir import eval_run

PREDICTOR_DICT = {
    'Pra': "precipitation_AWAP",
    'Ets': "et_morton_point_SILO",
    'Prs': "precipitation_SILO",
    'Eas': "et_morton_actual_SILO",
    'Ews': "et_morton_wet_SILO",
    'Ess': "et_short_crop_SILO",
    'Ecs': "et_tall_crop_SILO",
    'Els': "evap_morton_lake_SILO",
    'Eps': "evap_pan_SILO",
    'Eys': "evap_syn_SILO",
    'Soa': "solarrad_AWAP",
    'Tma': "tmax_AWAP",
    'Tia': "tmin_AWAP",
    'Vpa': "vprp_AWAP",
    'Mss': "mslp_SILO",
    'Ras': "radiation_SILO",
    'Rhs': "rh_tmax_SILO",
    'Ris': "rh_tmin_SILO",
    'Tms': "tmax_SILO",
    'Tis': "tmin_SILO",
    'Vps': "vp_SILO",
    'Vds': "vp_deficit_SILO",
}


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
    mapped_predictors = [PREDICTOR_DICT.get(predictor, None) for predictor in predictor_names]

    if None in mapped_predictors:
        missing_predictor = predictor_names[mapped_predictors.index(None)]
        error = f"Predictor '{missing_predictor}' not found in the dictionary."
        raise PredictorNotFoundError(error)

    return mapped_predictors


@contextmanager
def change_directory(new_path):
    """
    A context manager for temporarily changing the working directory.

    Parameters:
    - new_path (str): The path to change the working directory to.

    Usage:
    ```
    with change_directory('/path/to/your/directory'):
        # Your task goes here
        print("Current directory:", os.getcwd())
    # Now you're back to the original directory
    print("Current directory after the context manager:", os.getcwd())
    ```
    """
    # Get the current working directory
    original_path = os.getcwd()

    try:
        # Change the working directory to the desired path
        os.chdir(new_path)
        yield
    finally:
        # Revert back to the original directory, even if an exception occurs
        os.chdir(original_path)
