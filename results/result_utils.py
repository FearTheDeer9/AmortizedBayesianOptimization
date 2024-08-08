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
    noisy_suffix = r"\.pickle" if noiseless else r"noisy\.pickle"
    noisy_string = "" if noiseless else "_noisy"

    # Load and aggregate results for CEO
    ceo_string = rf".*_ceo_.*_results_{n_obs}_{n_int}_{noisy_suffix}"
    ceo_results = load_results(base_path, ceo_string)
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
        filename = f"{base_path}/{experiment}_{n_obs}_{n_int}_{noisy_string}"
        plt.savefig(filename)

    plt.show()


def all_means(
    base_path,
    num_cbo_graphs,
    n_obs=200,
    n_int=2,
    noiseless=True,
    graph_idxs=None,
    has_ceo: bool = True,
    has_bo: bool = True,
    has_cbo: bool = True,
):

    if graph_idxs is None:
        graph_idxs = range(num_cbo_graphs)

    mean_results = {}
    noisy_suffix = r"\.pickle" if noiseless else r"noisy\.pickle"
    noisy_string = "" if noiseless else "_noisy"

    # Load and aggregate results for CEO
    if has_ceo:
        ceo_string = rf".*_ceo_.*_results_{n_obs}_{n_int}_{noisy_suffix}"
        ceo_results = load_results(base_path, ceo_string)
        ceo_results = np.hstack(
            [ceo_result["Per_trial_Y"] for ceo_result in ceo_results]
        )
        ceo_mean = np.mean(ceo_results)
        ceo_std = np.std(ceo_results)
        mean_results["ceo"] = {"mean": ceo_mean, "std": ceo_std}

    # Load and aggregate results for BO
    # Load and aggregate results for CEO
    if has_bo:
        bo_string = rf".*_bo_results_{n_obs}_{n_int}_{noisy_suffix}"
        bo_results = load_results(base_path, bo_string)
        bo_results = np.hstack([bo_result["Per_trial_Y"] for bo_result in bo_results])
        bo_mean = np.mean(bo_results)
        bo_std = np.std(bo_results)
        mean_results["bo"] = {"mean": bo_mean, "std": bo_std}

    if has_cbo:
        for graph_index in range(num_cbo_graphs):
            cbo_string = (
                rf".*_cbo_results_{n_obs}_{n_int}_graph_{graph_index}{noisy_suffix}"
            )
            cbo_results = load_results(base_path, cbo_string)
            cbo_results = np.hstack(
                [cbo_result["Per_trial_Y"] for cbo_result in cbo_results]
            )
            cbo_mean = np.mean(cbo_results)
            cbo_std = np.std(cbo_results)
            mean_results[f"cbo_{graph_index}"] = {"mean": cbo_mean, "std": cbo_std}

    cbo_string = rf".*_cbo_unknown_results_{n_obs}_{n_int}{noisy_suffix}"
    cbo_results = load_results(base_path, cbo_string)
    cbo_results = np.hstack([cbo_result["Per_trial_Y"] for cbo_result in cbo_results])
    cbo_mean = np.mean(cbo_results)
    cbo_std = np.std(cbo_results)
    mean_results["cbo_unknown"] = {"mean": cbo_mean, "std": cbo_std}
    return mean_results


def all_best(
    base_path,
    num_cbo_graphs,
    n_obs=200,
    n_int=2,
    noiseless=True,
    graph_idxs=None,
    has_ceo: bool = True,
    has_bo: bool = True,
    has_cbo: bool = True,
):
    if graph_idxs is None:
        graph_idxs = range(num_cbo_graphs)

    min_results = {}
    noisy_suffix = r"\.pickle" if noiseless else r"noisy\.pickle"
    noisy_string = "" if noiseless else "_noisy"

    # Load and aggregate results for CEO
    if has_ceo:
        ceo_string = rf".*_ceo_.*_results_{n_obs}_{n_int}_{noisy_suffix}"
        ceo_results = load_results(base_path, ceo_string)
        ceo_results = np.vstack([ceo_result["Best_Y"] for ceo_result in ceo_results])
        ceo_best_results = ceo_results.min(axis=1)
        ceo_mean = np.mean(ceo_best_results)
        ceo_std = np.std(ceo_best_results)
        min_results["ceo"] = {"mean": ceo_mean, "std": ceo_std}

    # Load and aggregate results for BO
    # Load and aggregate results for CEO
    if has_bo:
        bo_string = rf".*_bo_results_{n_obs}_{n_int}_{noisy_suffix}"
        bo_results = load_results(base_path, bo_string)
        bo_results = np.vstack([bo_result["Best_Y"] for bo_result in bo_results])
        bo_best_results = bo_results.min(axis=1)
        bo_mean = np.mean(bo_best_results)
        bo_std = np.std(bo_best_results)
        min_results["bo"] = {"mean": bo_mean, "std": bo_std}

    if has_cbo:
        for graph_index in range(num_cbo_graphs):
            cbo_string = (
                rf".*_cbo_results_{n_obs}_{n_int}_graph_{graph_index}{noisy_suffix}"
            )
            cbo_results = load_results(base_path, cbo_string)
            cbo_results = np.vstack(
                [cbo_result["Best_Y"] for cbo_result in cbo_results]
            )
            cbo_best_results = cbo_results.min(axis=1)
            cbo_mean = np.mean(cbo_best_results)
            cbo_std = np.std(cbo_best_results)
            min_results[f"cbo_{graph_index}"] = {"mean": cbo_mean, "std": cbo_std}

    cbo_string = rf".*_cbo_unknown_results_{n_obs}_{n_int}{noisy_suffix}"
    cbo_results = load_results(base_path, cbo_string)
    cbo_results = np.hstack([cbo_result["Best_Y"] for cbo_result in cbo_results])
    cbo_mean = np.mean(cbo_results)
    cbo_std = np.std(cbo_results)
    min_results["cbo_unknown"] = {"mean": cbo_mean, "std": cbo_std}
    return min_results


# def all_best(
#     base_path, num_cbo_graphs, n_obs=200, n_int=2, noiseless=True, graph_idxs=None
# ):

#     min_results = {}
#     noisy_suffix = r"\.pickle" if noiseless else r"noisy\.pickle"
#     noisy_string = "" if noiseless else "_noisy"

#     # Load and aggregate results for CEO
#     ceo_string = rf".*_ceo_.*_results_{n_obs}_{n_int}_{noisy_suffix}"
#     ceo_results = load_results(base_path, ceo_string)
#     ceo_results = np.vstack([ceo_result["Best_Y"] for ceo_result in ceo_results])
#     ceo_best_results = ceo_results.min(axis=1)
#     ceo_mean = np.mean(ceo_best_results)
#     ceo_std = np.std(ceo_best_results)
#     min_results["ceo"] = {"mean": ceo_mean, "std": ceo_std}

#     # Load and aggregate results for BO
#     # Load and aggregate results for CEO
#     bo_string = rf".*_bo_results_{n_obs}_{n_int}_{noisy_suffix}"
#     bo_results = load_results(base_path, bo_string)
#     bo_results = np.vstack([bo_result["Best_Y"] for bo_result in bo_results])
#     bo_best_results = bo_results.min(axis=1)
#     bo_mean = np.mean(bo_best_results)
#     bo_std = np.std(bo_best_results)
#     min_results["bo"] = {"mean": bo_mean, "std": bo_std}

#     for graph_index in range(num_cbo_graphs):
#         cbo_string = (
#             rf".*_cbo_results_{n_obs}_{n_int}_graph_{graph_index}{noisy_suffix}"
#         )
#         cbo_results = load_results(base_path, cbo_string)
#         cbo_results = np.vstack([cbo_result["Best_Y"] for cbo_result in cbo_results])
#         cbo_best_results = cbo_results.min(axis=1)
#         cbo_mean = np.mean(cbo_best_results)
#         cbo_std = np.std(cbo_best_results)
#         min_results[f"cbo_{graph_index}"] = {"mean": cbo_mean, "std": cbo_std}

#     return min_results


def iterations_to_min(
    base_path, num_cbo_graphs, n_obs=100, n_int=2, noiseless=True, graph_idxs=None
):
    argmin_results = {}
    noisy_suffix = r"\.pickle" if noiseless else r"noisy\.pickle"
    noisy_string = "" if noiseless else "_noisy"

    # Load and aggregate results for CEO
    ceo_string = rf".*_ceo_.*_results_{n_obs}_{n_int}_{noisy_suffix}"
    ceo_results = load_results(base_path, ceo_string)
    ceo_results = np.vstack([ceo_result["Best_Y"] for ceo_result in ceo_results])
    ceo_best_results = ceo_results.argmin(axis=1)
    ceo_mean = np.mean(ceo_best_results)
    ceo_std = np.std(ceo_best_results)
    argmin_results["ceo"] = {"mean": ceo_mean, "std": ceo_std}

    # Load and aggregate results for BO
    # Load and aggregate results for CEO
    bo_string = rf".*_bo_results_{n_obs}_{n_int}_{noisy_suffix}"
    bo_results = load_results(base_path, bo_string)
    bo_results = np.vstack([bo_result["Best_Y"] for bo_result in bo_results])
    bo_best_results = bo_results.argmin(axis=1)
    bo_mean = np.mean(bo_best_results)
    bo_std = np.std(bo_best_results)
    argmin_results["bo"] = {"mean": bo_mean, "std": bo_std}

    for graph_index in range(num_cbo_graphs):
        cbo_string = (
            rf".*_cbo_results_{n_obs}_{n_int}_graph_{graph_index}{noisy_suffix}"
        )
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.vstack([cbo_result["Best_Y"] for cbo_result in cbo_results])
        cbo_best_results = cbo_results.argmin(axis=1)
        cbo_mean = np.mean(cbo_best_results)
        cbo_std = np.std(cbo_best_results)
        argmin_results[f"cbo_{graph_index}"] = {"mean": cbo_mean, "std": cbo_std}

    return argmin_results


def all_uncertainties(
    base_path, num_cbo_graphs, n_obs=100, n_int=2, noiseless=True, graph_idxs=None
):
    uncertanties = {}
    noisy_suffix = r"\.pickle" if noiseless else r"noisy\.pickle"
    noisy_string = "" if noiseless else "_noisy"

    # Load and aggregate results for BO
    # Load and aggregate results for CEO
    bo_string = rf".*_bo_uncertainties_{n_obs}_{n_int}_{noisy_suffix}"
    bo_results = load_results(base_path, bo_string)
    bo_results = np.vstack([bo_result["total"] for bo_result in bo_results])
    bo_mean = np.mean(bo_results)
    bo_std = np.std(bo_results)
    uncertanties["bo"] = {"mean": bo_mean, "std": bo_std}

    for graph_index in range(num_cbo_graphs):
        cbo_string = (
            rf".*_cbo_uncertainties_{n_obs}_{n_int}_graph_{graph_index}{noisy_suffix}"
        )
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.vstack([cbo_result["total"] for cbo_result in cbo_results])
        cbo_mean = np.mean(cbo_results)
        cbo_std = np.std(cbo_results)
        uncertanties[f"cbo_{graph_index}"] = {"mean": cbo_mean, "std": cbo_std}

    return uncertanties


def box_plots_mean(mean_results: dict, experiment: str):
    # Setup the plot
    fig, ax = plt.subplots()
    for i, (key, val) in enumerate(mean_results.items()):
        mean, std = val["mean"], val["std"]
        # Create the box
        box = plt.Rectangle(
            (i - 0.4, mean - std), 0.8, 2 * std, color="lightblue", alpha=0.5
        )
        ax.add_patch(box)
        # Plot the mean line
        plt.plot([i - 0.4, i + 0.4], [mean, mean], color="red")

    # Set plot properties
    plt.xticks(range(len(mean_results)), labels=mean_results.keys(), rotation=45)
    plt.ylabel("Value")
    plt.title(f"{experiment} y value for each algorithm")
    plt.grid(True)
    plt.show()
