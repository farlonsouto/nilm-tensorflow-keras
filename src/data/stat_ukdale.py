import json

import pandas as pd
from nilmtk import DataSet

from src import hyper_params


def compute_stats(appliance_list):
    """
    Computes statistical data (mean, median, std, etc.) for active power, apparent power,
    and specified appliances across buildings.
    """

    dataset = DataSet('../../datasets/ukdale.h5')

    active_power_building = {}
    apparent_power_building = {}
    appliance_power_building = {}

    all_active_power_global = []
    all_apparent_power_global = []
    appliance_power_global = {appliance: [] for appliance in appliance_list}

    for building in dataset.buildings:
        train_elec = dataset.buildings[building].elec

        # Handle mains power statistics
        train_mains = train_elec.mains()
        mains_data_frame = train_mains.load()

        all_active_power = []
        all_apparent_power = []

        for train_mains_df in mains_data_frame:
            # Check if active power is available
            if ('power', 'active') in train_mains_df.columns:
                active_power = train_mains_df[('power', 'active')]
                all_active_power.append(active_power)
                all_active_power_global.append(active_power)

            # Check if apparent power is available
            if ('power', 'apparent') in train_mains_df.columns:
                apparent_power = train_mains_df[('power', 'apparent')]
                all_apparent_power.append(apparent_power)
                all_apparent_power_global.append(apparent_power)

        # Compute stats for active power
        if all_active_power:
            combined_active_power = pd.concat(all_active_power, axis=0)

            active_power_building[str(building)] = {
                "mean": float(combined_active_power.mean()),
                "median": float(combined_active_power.median()),
                "std": float(combined_active_power.std()),
                "Quantiles": str(combined_active_power.quantile([.25, .5, .75]).values)
            }

        # Compute stats for apparent power
        if all_apparent_power:
            combined_apparent_power = pd.concat(all_apparent_power, axis=0)
            apparent_power_building[str(building)] = {
                "mean": float(combined_apparent_power.mean()),
                "median": float(combined_apparent_power.median()),
                "std": float(combined_apparent_power.std()),
                "Quantiles": str(combined_apparent_power.quantile([.25, .5, .75]).values)
            }

        # Handle appliances
        appliance_power_building[str(building)] = {}
        for current_appliance in appliance_list:
            nilmtk_appliance = None
            try:
                nilmtk_appliance = train_elec[current_appliance]
            except KeyError:
                appliance_power_building[str(building)][current_appliance] = 'Not available'
                continue
            if nilmtk_appliance:
                appliance_data = nilmtk_appliance.load()
                all_appliance_power = []

                for appliance_df in appliance_data:
                    if ('power', 'active') in appliance_df.columns:
                        appliance_power = appliance_df[('power', 'active')]
                        all_appliance_power.append(appliance_power)
                        appliance_power_global[current_appliance].append(appliance_power)

                if all_appliance_power:
                    combined_appliance_power = pd.concat(all_appliance_power, axis=0)
                    appliance_power_building[str(building)][current_appliance] = {
                        "mean": float(combined_appliance_power.mean()),
                        "median": float(combined_appliance_power.median()),
                        "std": float(combined_appliance_power.std()),
                        "Quantiles": str(combined_appliance_power.quantile([.25, .5, .75]).values)
                    }

    # Compute global stats for all buildings
    the_entire_dataset_stats = {}

    # Active Power Global Stats
    if all_active_power_global:
        global_active_power = pd.concat(all_active_power_global, axis=0)
        the_entire_dataset_stats['active_power'] = {
            "mean": float(global_active_power.mean()),
            "std": float(global_active_power.std())
        }

    # Apparent Power Global Stats
    if all_apparent_power_global:
        global_apparent_power = pd.concat(all_apparent_power_global, axis=0)
        the_entire_dataset_stats['apparent_power'] = {
            "mean": float(global_apparent_power.mean()),
            "std": float(global_apparent_power.std())
        }

    # Appliance Global Stats
    the_entire_dataset_stats['appliance_power'] = {}
    for current_appliance, powers in appliance_power_global.items():
        if powers:
            config = hyper_params.for_model_appliance(current_appliance)
            on_threshold = config["on_threshold"]
            min_on_duration = config["min_on_duration"]
            global_appliance_power = pd.concat(powers, axis=0)
            mean_on_duration, std_on_duration = calculate_activation_stats(global_appliance_power, on_threshold,
                                                                           min_on_duration)
            the_entire_dataset_stats['appliance_power'][current_appliance] = {
                "mean": float(global_appliance_power.mean()),
                "std": float(global_appliance_power.std()),
                "ON_mean": float(
                    global_appliance_power[global_appliance_power > on_threshold].mean()),
                "ON_std": float(
                    global_appliance_power[global_appliance_power > on_threshold].std()),
                "ON_duration_mean": mean_on_duration,
                "ON_duration_std": std_on_duration
            }

    # Return the computed dictionaries
    return active_power_building, apparent_power_building, appliance_power_building, the_entire_dataset_stats


def calculate_activation_stats(appliance_power, on_threshold, min_on_duration) -> tuple:
    """
    Calculate statistics for the appliance ON durations.

    Parameters:
    - appliance_power: pd.Series containing appliance power data.
    - on_threshold: minimum value from which the appliance is considered ON
    - min_on_min_on_duration: the minimum amount of seconds the appliance must be active to be considered a ON

    Returns:
    - A tuple of ( mean_on_duration, std_on_duration)
    """
    # Filter ON periods
    on_values = appliance_power[appliance_power > on_threshold]

    # Identify contiguous ON periods
    # Create a boolean mask for ON periods
    on_mask = appliance_power > on_threshold

    # Use cumsum to assign a unique label to each contiguous ON period
    on_periods = (on_mask != on_mask.shift()).cumsum() * on_mask

    # Group by the period labels and calculate their durations (number of time steps)
    on_durations = on_periods.value_counts().loc[lambda x: x > min_on_duration].values

    # Non-active periods make the largest label count, so we remove it
    on_durations.sort()
    on_durations = on_durations[:-1]

    # Calculate mean and std for ON durations
    mean_on_duration = on_durations.mean() if len(on_durations) > 0 else 0
    std_on_duration = on_durations.std() if len(on_durations) > 0 else 0

    return mean_on_duration, std_on_duration


active_building, apparent_building, appliance_building, global_stats = compute_stats(
    ['fridge', 'kettle', 'washer', 'microwave', 'dish washer'])

print("Aggregated Active Power Stats per building with Quantiles [25%, 50%, 75%]:")
print(json.dumps(active_building, indent=4))
print("Aggregated Apparent Power Stats per building  with Quantiles [25%, 50%, 75%]:")
print(json.dumps(apparent_building, indent=4))
print("Appliance Active Power Stats per building  with Quantiles [25%, 50%, 75%]:")
print(json.dumps(appliance_building, indent=4))

print("Global Stats Across All Buildings:")
print(json.dumps(global_stats, indent=4))
