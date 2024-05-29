import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np


# Function to load all results of a given type
def load_results(base_path, regex):
    results = []
    for filename in os.listdir(base_path):
        if bool(re.match(regex, filename)):
            with open(os.path.join(base_path, filename), "rb") as file:
                results.append(pickle.load(file))
    return results


# Function to compute statistics across runs
def aggregate_results(results, key):
    all_data = [res[key] for res in results if key in res]
    mean_data = np.mean(all_data, axis=0)
    std_dev_data = np.std(all_data, axis=0)
    return mean_data, std_dev_data


def plot_everything(
    base_path,
    experiment,
    num_cbo_graphs,
    n_obs=100,
    n_int=2,
    noiseless=True,
    save_file=False,
    plot_ceo=True,
    plot_cbo=True,
    plot_bo=True,
    graph_idxs=None,
):
    if graph_idxs is None:
        graph_idxs = range(num_cbo_graphs)
    # Determine the string suffix based on the noiseless flag
    noisy_suffix = r"\.pickle" if noiseless else r"_noisy\.pickle"
    noisy_string = "" if noiseless else "_noisy"

    # Load and aggregate results for CEO
    ceo_string = f".*_ceo_.*results{noisy_suffix}"
    ceo_results = load_results(base_path, r".*_ceo.*results" + noisy_suffix)
    ceo_mean, ceo_std = aggregate_results(ceo_results, experiment)

    # Load and aggregate results for BO
    # Load and aggregate results for CEO
    bo_string = rf".*_bo_results_{n_obs}_{n_int}_{noisy_suffix}"
    bo_results = load_results(base_path, bo_string)
    bo_mean, bo_std = aggregate_results(bo_results, experiment)

    # Initialize plot for CEO results
    if plot_ceo:
        x_values = range(len(ceo_mean))
        plt.fill_between(x_values, ceo_mean - ceo_std, ceo_mean + ceo_std, alpha=0.2)
        plt.plot(x_values, ceo_mean, label="CEO Mean")

    if plot_bo:
        x_values = range(len(bo_mean))
        plt.fill_between(x_values, bo_mean - bo_std, bo_mean + bo_std, alpha=0.2)
        plt.plot(x_values, bo_mean, label="BO Mean")

    # Process each CBO graph dynamically
    for graph_index in range(num_cbo_graphs):
        cbo_string = (
            rf".*_cbo_results_{n_obs}_{n_int}_graph_{graph_index}{noisy_suffix}"
        )
        cbo_results = load_results(base_path, cbo_string)
        cbo_mean, cbo_std = aggregate_results(cbo_results, experiment)

        # Plotting results for each CBO graph
        if graph_index in graph_idxs:
            x_values = range(len(cbo_mean))
            plt.fill_between(
                x_values, cbo_mean - cbo_std, cbo_mean + cbo_std, alpha=0.2
            )
            plt.plot(x_values, cbo_mean, label=f"CBO Graph {graph_index}")

    # Final plot adjustments
    plt.xlabel("Trial")
    plt.ylabel("Y value")
    plt.legend()

    if save_file:
        filename = f"{base_path}/{experiment}{noisy_string}"
        plt.savefig(filename)

    plt.show()
