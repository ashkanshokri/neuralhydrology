import os
import re
from collections import defaultdict
from contextlib import contextmanager
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import xarray as xr
from tqdm import tqdm

from neuralhydrology.evaluation.metrics import calculate_metrics
from neuralhydrology.utils.config import Config
from neuralhydrology.utils.errors import AllNaNError
from neuralhydrology.utils.nh_results_ensemble_ash import \
    create_results_ensemble

ATTR_FILE = '/Users/sho108/Desktop/z/Data/CAMELS_AUS/CAMELS_AUS_Attributes&Indices_MasterTable.csv'

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


def generate_configs(config_file: Path,
                     static_attributes: List[str],
                     dynamic_predictor_options: List[str],
                     output_dir: Path = None,
                     date_ranges: Dict[str, List[Tuple[str, str]]] = None,
                     folds=2,
                     basin_files: Dict[str, List[Path]] = None,
                     prefix: str = 'test',
                     other_configs: Dict[str, Union[str, int, float, Path]] = {}):
    """
    Generate configurations for experiments.

    Args:
        config_file (Path): Path to the configuration file.
        static_attributes (List[str]): List of static attributes.
        dynamic_predictor_options (List[str]): List of dynamic predictor options.
        output_dir (Path, optional): Output directory to store generated configurations. Defaults to None.
        date_ranges (Dict[str, List[Tuple[str, str]]], optional): Dictionary containing date ranges for train, validation, and test sets. Defaults to None.
        basin_files (Dict[str, List[Path]], optional): Dictionary containing basin files for train, validation, and test sets. Defaults to None.
        prefix (str, optional): Prefix to be used in experiment names. Defaults to 'test'.
        other_configs (Dict[str, Union[str, int, float, Path]], optional): Other configurations to be included. Defaults to {}.
    """
    if output_dir:
        output_dir.mkdir(exist_ok=True, parents=True)

    for static_attrs in static_attributes:
        for predictors in dynamic_predictor_options:
            for fold in range(folds):
                cfg = Config(config_file)

                cfg.update_config({
                    'dynamic_inputs': get_predictor_list(predictors),
                    'static_attributes': get_static_attr_list(static_attrs),
                    'experiment_name': f'{prefix}_{fold}_{predictors}_{static_attrs}',
                })
                if basin_files:
                    train_basin_file = basin_files['train'][fold]
                    validation_basin_file = basin_files['validation'][fold]
                    test_basin_file = basin_files['test'][fold]

                    cfg.update_config({
                        'train_basin_file': train_basin_file,
                        'test_basin_file': test_basin_file,
                        'validation_basin_file': validation_basin_file,
                    })

                if date_ranges:
                    train_start_date, train_end_date = date_ranges['train'][fold]
                    validation_start_date, validation_end_date = date_ranges['validation'][fold]
                    test_start_date, test_end_date = date_ranges['test'][fold]

                    cfg.update_config({
                        'train_start_date': train_start_date,
                        'train_end_date': train_end_date,
                        'validation_start_date': validation_start_date,
                        'validation_end_date': validation_end_date,
                        'test_start_date': test_start_date,
                        'test_end_date': test_end_date,
                    })
                cfg.update_config(other_configs)

                # Start training
                cfg.dump_config(output_dir, cfg.experiment_name + '.yml')


def get_freqs(ens):
    k = list(ens.keys())[0]
    return list(ens[k].keys())


def get_basins(ens):
    return ens.keys()


def join_spatial(results):
    combined = {}
    combined = defaultdict(lambda: defaultdict(dict))
    frequencies = get_freqs(results[0])
    for ens in results:
        for basin in ens:
            for freq in frequencies:
                combined[basin][freq] = ens[basin][freq]['xr']

    return combined


def join_temporal(results):
    frequencies = get_freqs(results[0])
    basins = get_basins(results[0])
    combined = defaultdict(lambda: defaultdict(dict))
    for basin in basins:
        for freq in frequencies:
            ds = xr.concat([e[basin][freq]['xr'] for e in results], dim='datetime')
            ds = ds.rename({'datetime': 'date'})
            combined[basin][freq] = ds
    return combined


def join(results, method):
    if method == 'temporal':
        return join_temporal(results)
    elif method == 'spatial':
        return join_spatial(results)


def get_ensemble_metric(combined_ensemble, target_var='streamflow_mmd', metric='NSE', freq='1D'):

    results = defaultdict(dict)
    basins = get_basins(combined_ensemble)

    for basin in tqdm(basins):
        ds = combined_ensemble[basin][freq]
        for i, member in enumerate(ds['member']):
            member_obs = ds[f'{target_var}_obs'].sel(member=member)
            member_sim = ds[f'{target_var}_sim'].sel(member=member)
            try:
                results[i][basin] = calculate_metrics(member_obs, member_sim, metrics=[metric], resolution=freq)
            except AllNaNError:
                results[i][basin] = np.nan

    df = pd.concat([pd.DataFrame.from_dict(r) for r in results.values()]).T
    #df.columns = ds['member']

    return df


def all_folds(pattern, parent, folds=2, **kwargs):
    ens_res_list = []
    for fold in range(folds):
        try:
            run_dir_pattern = pattern.format(fold=fold)
            run_dirs = list(parent.glob(run_dir_pattern))
            print(run_dir_pattern)
            ens = create_results_ensemble(run_dirs, **kwargs)
            ens_res_list.append(ens)
        except ValueError:
            pass

    return ens_res_list


def combine(df, attr_file):
    # Load attribute data
    attr_df = pd.read_csv(attr_file, index_col=0)

    # Combine metric and attribute data
    data = pd.concat([df, attr_df], axis=1).reset_index()
    return data


# plots:
def get_data_melt(df, x, attr_file):

    data = combine(df, attr_file)

    # Melt the data for better visualization
    return data.melt(value_vars=df.columns, id_vars=x)  #['drainage_division', 'river_region', 'state_outlet']


def _boxplot(all_metrics_df, ax=None, x='state_outlet', attr_file=ATTR_FILE):
    data_melted = get_data_melt(all_metrics_df, x, attr_file)
    # Set up the plotting environment
    plt.figure(figsize=(15, 5))

    # Create a boxplot and stripplot using seaborn
    hue_order = None  # all_metrics_df.median().sort_values(ascending=False).keys()

    if not ax:
        fig, ax = plt.subplots()

    sns.boxplot(data_melted, y='value', x=x, hue='variable', hue_order=hue_order, saturation=0.3, gap=0.1, ax=ax)
    sns.stripplot(data=data_melted, y='value', x=x, hue='variable', dodge=True, ax=ax, legend=False)

    # Set y-axis limits
    ax.set_ylim(-2, 1)

    # Move legend to a more suitable position
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    ax.set_ylabel('nse')


def _fdc(all_metrics_df, ax=None, x='state_outlet', color=None, attr_file=ATTR_FILE):
    data_melted = get_data_melt(all_metrics_df, x, attr_file)

    hue_order = None
    if not ax:
        fig, ax = plt.subplots()
    sns.ecdfplot(data_melted,
                 y='value',
                 hue='variable',
                 hue_order=hue_order,
                 complementary=True,
                 alpha=0.7,
                 ax=ax,
                 color=color)
    ax.set_ylim(-1, 1)
    sns.move_legend(
        ax,
        "lower center",
        bbox_to_anchor=(.5, 1),
        ncol=2,
        title=None,
        frameon=False,
    )
    ax.set_ylabel('nse')


def _scatter(all_metrics_df, ax=None, x=None, y=None, hue='state_outlet', attr_file=ATTR_FILE):
    data = combine(all_metrics_df, attr_file)
    # Set up the plotting environment
    plt.figure(figsize=(15, 5))

    # Create a boxplot and stripplot using seaborn
    hue_order = None  # all_metrics_df.median().sort_values(ascending=False).keys()

    if not ax:
        fig, ax = plt.subplots()

    #sns.boxplot(data_melted, y='value', x=x, hue='variable', hue_order=hue_order, saturation=0.3, gap=0.1, ax=ax)
    #sns.stripplot(data=data_melted, y='value', x=x, hue='variable', dodge=True, ax=ax, legend=False)
    sns.scatterplot(data=data, y=y, x=x, hue=hue, s=7)
    plt.plot([0, 1], [0, 1], ':r')
    plt.ylim([0, 1])
    plt.xlim([0, 1])
    plt.gca().set_aspect('equal')

    # Move legend to a more suitable position
    sns.move_legend(ax, "lower center", bbox_to_anchor=(.5, 1), ncol=2, title=None, frameon=False)
    ax.set_ylabel(y)
    ax.set_xlabel(x)


def evaluate_experiments(experiment_configs, memory='max', base_path=None, folds=2):
    """
    Evaluate multiple experiments and generate metrics.

    Args:
        experiment_configs (list): List of dictionaries containing experiment configurations.
        memory (str or int, optional): Memory parameter. 'max' for maximum memory, integer for specific memory index. Defaults to 'max'.
        base_path (str, optional): Base path for experiment data. Defaults to None.

    Returns:
        pd.DataFrame: DataFrame containing evaluation metrics for each experiment.
    """
    metrics_dict = {}

    for exp in experiment_configs:

        print(exp)
        ens_list = all_folds(exp['pattern'], base_path, metrics=['NSE'], folds=folds)
        print('-' * 10, len(ens_list))
        combined_ensemble = join(ens_list, exp['method'])
        df = get_ensemble_metric(combined_ensemble, target_var='streamflow_mmd', metric='NSE', freq='1D')
        name = exp.get('name', exp['pattern'])

        if isinstance(memory, str):
            if memory == 'max':
                metrics_dict[name] = df.iloc[:, df.median().argmax()]
            else:
                raise NotImplementedError("Memory option not implemented.")
        else:
            metrics_dict[name] = df.iloc[:, memory]

    all_metrics = pd.DataFrame.from_dict(metrics_dict)

    _, (ax1, ax2) = plt.subplots(ncols=2, figsize=(15, 7))
    _boxplot(all_metrics, ax=ax1)
    _fdc(all_metrics, ax=ax2)

    return all_metrics
