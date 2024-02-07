import os
import re
from contextlib import contextmanager
from typing import Dict, Tuple, Union

import pandas as pd
import xarray as xr

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

STATIC_ATTR_DICT = {
    'Are': 'catchment_area',
    'Pme': 'p_mean',
    'Pet': 'pet_mean',
    'Ari': 'aridity',
    'Pse': 'p_seasonality',
    'Fra': 'frac_snow',
    'Hpf': 'high_prec_freq',
    'Hpd': 'high_prec_dur',
    'Hig': 'high_prec_timing',
    'Lpf': 'low_prec_freq',
    'Lpd': 'low_prec_dur',
    'Lpt': 'low_prec_timing',
    'Qme': 'q_mean',
    'Run': 'runoff_ratio',
    'Str': 'stream_elas',
    'Slo': 'slope_fdc',
    'Bas': 'baseflow_index',
    'hdf': 'hdf_mean',
    'Q5': 'Q5',
    'Q95': 'Q95',
    'Hqf': 'high_q_freq',
    'Hqd': 'high_q_dur',
    'Lqf': 'low_q_freq',
    'Lqd': 'low_q_dur',
    'Zer': 'zero_q_freq',
    'Geo': 'geol_prim',
    'Gpp': 'geol_prim_prop',
    'Gs': 'geol_sec',
    'Gsp': 'geol_sec_prop',
    'Unc': 'unconsoldted',
    'Ign': 'igneous',
    'Sil': 'silicsed',
    'Car': 'carbnatesed',
    'Oth': 'othersed',
    'Met': 'metamorph',
    'Sed': 'sedvolc',
    'Old': 'oldrock',
    'Cla': 'claya',
    'Clb': 'clayb',
    'San': 'sanda',
    'Sot': 'solum_thickness',
    'Ksa': 'ksat',
    'Sol': 'solpawhc',
    'Emi': 'elev_min',
    'Ema': 'elev_max',
    'Eme': 'elev_mean',
    'Era': 'elev_range',
    'Mea': 'mean_slope_pct',
    'Ups': 'upsdist',
    'Std': 'strdensity',
    'Sta': 'strahler',
    'Elo': 'elongratio',
    'Rel': 'relief',
    'Ref': 'reliefratio',
}


class PredictorNotFoundError(Exception):
    pass


def get_static_attr_list(s: str) -> str:
    """
    Extracts predictor names from the input string and maps them to corresponding values using STATIC_ATTR_DICT.

    Parameters:
    - s (str): Input string containing predictor abbreviations.

    Returns:
    - str: Concatenated string of mapped predictor values.
    """
    return get_mapped_values(s, STATIC_ATTR_DICT)


def get_predictor_list(s: str) -> str:
    """
    Extracts predictor names from the input string and maps them to corresponding values using PREDICTOR_DICT.

    Parameters:
    - s (str): Input string containing predictor abbreviations.

    Returns:
    - str: Concatenated string of mapped predictor values.
    """
    return get_mapped_values(s, PREDICTOR_DICT)


def get_mapped_values(s: str, dictionary: dict) -> str:
    """
    Extracts predictor names from the input string and maps them to corresponding values using the provided dictionary.

    Parameters:
    - s (str): Input string containing predictor abbreviations.
    - dictionary (dict): Dictionary mapping predictor abbreviations to values.

    Returns:
    - str: Concatenated string of mapped predictor values.
    """

    predictor_names = re.findall(r'[A-Z][a-z]*', s)
    mapped_predictors = [dictionary.get(predictor, None) for predictor in predictor_names]

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


def merge_basins(result: Dict[str, Dict[str, Union[str, pd.DataFrame]]]) -> xr.Dataset:
    """
    Merge basin datasets into a single xarray Dataset.

    Parameters
    ----------
    result : Dict[str, Dict[str, Union[str, pd.DataFrame]]]
        Dictionary containing basin-specific results.

    Returns
    -------
    xr.Dataset
        Combined xarray Dataset.
    """
    basins = list(result.keys())
    data_values = [v['1D']['xr'] for v in result.values()]
    combined_dataset = xr.concat(data_values, dim='basin').assign_coords(basin=basins)
    return combined_dataset


def combine_folds(results: Dict[int, Dict[str, Dict[str, Union[str, pd.DataFrame]]]], dim_name: str) -> pd.DataFrame:
    """
    Combine datasets from multiple folds along the specified dimension.

    Parameters
    ----------
    results : Dict[int, Dict[str, Dict[str, Union[str, pd.DataFrame]]]]
        Dictionary containing results for different folds.
    dim_name : str
        Name of the dimension along which datasets will be concatenated.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame.
    """
    fold_keys = results.keys()
    merged_datasets = [merge_basins(results[fold]) for fold in fold_keys]
    combined_dataset = xr.concat(merged_datasets, dim=dim_name)
    return combined_dataset.to_dataframe()


def combine_spatial_folds(results: Dict[int, Dict[str, Dict[str, Union[str, pd.DataFrame]]]]) -> pd.DataFrame:
    """
    Combine datasets from multiple spatial folds.

    Parameters
    ----------
    results : Dict[int, Dict[str, Dict[str, Union[str, pd.DataFrame]]]]
        Dictionary containing results for different folds.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame.
    """
    return combine_folds(results, dim_name='basin')


def combine_temporal_folds(results: Dict[int, Dict[str, Dict[str, Union[str, pd.DataFrame]]]]) -> pd.DataFrame:
    """
    Combine datasets from multiple temporal folds.

    Parameters
    ----------
    results : Dict[int, Dict[str, Dict[str, Union[str, pd.DataFrame]]]]
        Dictionary containing results for different folds.

    Returns
    -------
    pd.DataFrame
        Combined DataFrame.
    """
    return combine_folds(results, dim_name='date')


def create_pivot_tables(df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Create pivot tables for simulated and observed streamflow.

    Parameters
    ----------
    df : pd.DataFrame
        DataFrame containing the streamflow data.

    Returns
    -------
    Tuple[pd.DataFrame, pd.DataFrame]
        Tuple containing pivot tables for simulated and observed streamflow.
    """
    pivot_table_simulated = df.pivot_table(values='streamflow_mmd_sim', index='date', columns='basin')
    pivot_table_observed = df.pivot_table(values='streamflow_mmd_obs', index='date', columns='basin')
    return pivot_table_simulated, pivot_table_observed


def calculate_metric(df_sim: pd.DataFrame, df_obs: pd.DataFrame, metric, name='metric') -> pd.DataFrame:
    """
    Calculate a performance metric for simulated and observed streamflow.

    Parameters
    ----------
    df_sim : pd.DataFrame
        DataFrame containing simulated streamflow data.
    df_obs : pd.DataFrame
        DataFrame containing observed streamflow data.
    metric : function
        Metric function to be applied to the data.
    name : str, optional
        Name of the metric, by default 'metric'.

    Returns
    -------
    pd.DataFrame
        DataFrame containing the calculated metric for each basin.
    """
    metric_dict = {
        catchment: {
            name: metric(df_sim.loc[df_obs.index, catchment], df_obs[catchment])
        } for catchment in df_obs.columns
    }
    return pd.DataFrame.from_dict(metric_dict).T
