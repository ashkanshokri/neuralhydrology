from typing import Union, Dict

import geopandas as gpd
from pathlib import Path
from neuralhydrology.evaluation.utils import metrics_to_dataframe
import pandas as pd
from matplotlib import pyplot as plt


def spatial_plot(shpfile: Union[str, Path],
                 results: Dict[str, Dict[str, pd.DataFrame]],
                 shp_basin_field: str,
                 metric: str = 'NSE',
                 **kwargs) -> None:
    """
    Visualize the results of a neural hydrology model evaluation on a map.

    Parameters:
    - shpfile (Union[str, Path]): Path to the shapefile containing basin boundaries.
    - results (Dict[str, Dict[str, pd.DataFrame]]): Output of the neural hydrology model evaluation.
    - shp_basin_field (str): Field in the shapefile representing basin IDs (default is 'CatchID').
    - metric (str): The evaluation metric to visualize (default is 'NSE').
    - **kwargs: Additional keyword arguments for the plot function.

    Returns:
    None
    """
    # Set default values for kwargs
    default_kwargs = dict(cmap='bwr_r', vmin=-1, vmax=1, legend=True, edgecolor='k', linewidth=0.3)
    kwargs = {**default_kwargs, **kwargs}

    # Read the shapefile
    gdf = gpd.read_file(shpfile)

    # Extract the metric (e.g., 'NSE') dataframe from the results dictionary
    metric_df = metrics_to_dataframe(results, metrics=[metric])

    if not metric_df.empty:
        # Merge the shapefile with the metric dataframe
        gdf_merged = pd.merge(gdf, metric_df, left_on=shp_basin_field, right_index=True)

        # Plot the results on the map
        gdf_merged.plot(column=metric, **kwargs)
    else:
        print("Metric dataframe is empty. Cannot visualize results.")
