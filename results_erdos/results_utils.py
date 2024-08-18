import os
import pickle
import re

import matplotlib.cm as cm
import matplotlib.pyplot as plt
import numpy as np

base_path = "/Users/jeandurand/Documents/Masters Thesis/causal_bayes_opt/results/Erdos"


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
    n_obs=200,
    n_int=2,
    nonlinear=False,
    save_file=False,
    plot_all=True,
    plot_dr1=True,
    plot_dr2=True,
    plot_cbo=True,
    plot_random=True,
):
    inferno = cm.get_cmap("plasma")
    colors = inferno(np.linspace(0, 1, 5))
    nonlinear_string = "_nonlinear" if nonlinear else ""
    # colors = [
    #     "#1b9e77",  # Teal
    #     "#d95f02",  # Orange
    #     "#7570b3",  # Purple
    #     "#e7298a",  # Pink
    #     # "#66a61e",  # Green
    #     "#e6ab02",  # Yellow
    #     "#a6761d",  # Brown
    #     "#666666",  # Gray
    #     "#8c564b",  # Brownish Red
    #     # "#2ca02c",  # Green
    #     "#ff7f0e",  # Orange
    #     "#1f77b4",  # Blue
    #     "#aec7e8",  # Black
    #     "#ffbb78",  # Light Orange
    #     # "#98df8a",  # Light Green
    # ]

    # Initialize plot for CEO results
    if plot_all:
        # Load and aggregate results for CEO
        all_string = (
            rf".*_cbo_unknown_results_all_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        all_results = load_results(base_path, all_string)
        all_mean, all_std = aggregate_results(all_results, experiment)
        x_values = range(len(all_mean))
        plt.fill_between(
            x_values, all_mean - all_std, all_mean + all_std, alpha=0.2, color=colors[0]
        )
        plt.plot(x_values, all_mean, label="All", color=colors[0])

    if plot_dr1:
        # Load and aggregate results for CEO
        dr1_string = (
            rf".*_cbo_unknown_dr1_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr1_results = load_results(base_path, dr1_string)
        dr1_mean, dr1_std = aggregate_results(dr1_results, experiment)
        x_values = range(len(dr1_mean))
        plt.fill_between(
            x_values, dr1_mean - dr1_std, dr1_mean + dr1_std, alpha=0.2, color=colors[1]
        )
        plt.plot(x_values, dr1_mean, label="DR1", color=colors[1])

    if plot_dr2:
        # Load and aggregate results for CEO
        dr2_string = (
            rf".*_cbo_unknown_dr2_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr2_results = load_results(base_path, dr2_string)
        dr2_mean, dr2_std = aggregate_results(dr2_results, experiment)
        x_values = range(len(dr2_mean))
        plt.fill_between(
            x_values, dr2_mean - dr2_std, dr2_mean + dr2_std, alpha=0.2, color=colors[2]
        )
        plt.plot(x_values, dr2_mean, label="DR2", color=colors[2])

    if plot_cbo:
        # Load and aggregate results for CEO
        cbo_string = rf".*_cbo_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results = load_results(base_path, cbo_string)
        cbo_mean, cbo_std = aggregate_results(cbo_results, experiment)
        x_values = range(len(cbo_mean))
        plt.fill_between(
            x_values, cbo_mean - cbo_std, cbo_mean + cbo_std, alpha=0.2, color=colors[2]
        )
        plt.plot(x_values, cbo_mean, label="CBO", color=colors[3])

    if plot_random:
        # Load and aggregate results for CEO
        random_string = (
            rf".*_cbo_results_random_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        random_results = load_results(base_path, random_string)
        random_mean, random_std = aggregate_results(random_results, experiment)
        x_values = range(len(random_mean))
        plt.fill_between(
            x_values,
            random_mean - random_std,
            random_mean + random_std,
            alpha=0.2,
            color=colors[2],
        )
        plt.plot(x_values, random_mean, label="Random", color=colors[4])

    # Final plot adjustments
    plt.xlabel("Trial")
    plt.ylabel("Y value")
    plt.grid(True)
    plt.legend()

    if save_file:
        filename = f"{base_path}/{experiment}_{n_obs}_{n_int}"
        plt.savefig(filename)

    plt.show()


def all_means(
    base_path,
    n_obs=200,
    n_int=2,
    nonlinear: bool = False,
    has_all: bool = True,
    has_dr1: bool = True,
    has_dr2: bool = True,
    has_cbo: bool = True,
    has_random: bool = True,
):

    mean_results = {}
    nonlinear_string = "_nonlinear" if nonlinear else ""

    if has_all:
        all_string = (
            rf".*_cbo_unknown_results_all_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        all_results = load_results(base_path, all_string)
        all_results = np.hstack(
            [all_result["Per_trial_Y"] for all_result in all_results]
        )
        all_mean = np.mean(all_results)
        all_std = np.std(all_results)
        mean_results["all"] = {"mean": all_mean, "std": all_std}

    if has_dr1:
        dr1_string = (
            rf".*_cbo_unknown_dr1_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr1_results = load_results(base_path, dr1_string)
        dr1_results = np.hstack(
            [dr1_result["Per_trial_Y"] for dr1_result in dr1_results]
        )
        dr1_mean = np.mean(dr1_results)
        dr1_std = np.std(dr1_results)
        mean_results["dr1"] = {"mean": dr1_mean, "std": dr1_std}

    if has_dr2:
        dr2_string = (
            rf".*_cbo_unknown_dr2_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr2_results = load_results(base_path, dr2_string)
        dr2_results = np.hstack(
            [dr2_result["Per_trial_Y"] for dr2_result in dr2_results]
        )
        dr2_mean = np.mean(dr2_results)
        dr2_std = np.std(dr2_results)
        mean_results["dr2"] = {"mean": dr2_mean, "std": dr2_std}

    if has_cbo:
        cbo_string = rf".*_cbo_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.hstack(
            [cbo_result["Per_trial_Y"] for cbo_result in cbo_results]
        )
        cbo_mean = np.mean(cbo_results)
        cbo_std = np.std(cbo_results)
        mean_results["cbo"] = {"mean": cbo_mean, "std": cbo_std}

    if has_random:
        random_string = (
            rf".*_cbo_results_random_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        random_results = load_results(base_path, random_string)
        random_results = np.hstack(
            [random_result["Per_trial_Y"] for random_result in random_results]
        )
        random_mean = np.mean(random_results)
        random_std = np.std(random_results)
        mean_results["random"] = {"mean": random_mean, "std": random_std}

    return mean_results


def all_best(
    base_path,
    n_obs=200,
    n_int=2,
    nonlinear: bool = False,
    has_all: bool = True,
    has_dr1: bool = True,
    has_dr2: bool = True,
    has_cbo: bool = True,
    has_random: bool = True,
):

    min_results = {}
    nonlinear_string = "_nonlinear" if nonlinear else ""

    if has_all:
        all_string = (
            rf".*_cbo_unknown_results_all_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        all_results = load_results(base_path, all_string)
        all_results = np.vstack([all_result["Best_Y"] for all_result in all_results])
        all_best_results = all_results.min(axis=1)
        all_mean = np.mean(all_best_results)
        all_std = np.std(all_best_results)
        min_results["all"] = {"mean": all_mean, "std": all_std}

    if has_dr1:
        dr1_string = (
            rf".*_cbo_unknown_dr1_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr1_results = load_results(base_path, dr1_string)
        dr1_results = np.vstack([dr1_result["Best_Y"] for dr1_result in dr1_results])
        dr1_best_results = dr1_results.min(axis=1)
        dr1_mean = np.mean(dr1_best_results)
        dr1_std = np.std(dr1_best_results)
        min_results["dr1"] = {"mean": dr1_mean, "std": dr1_std}

    if has_dr2:
        dr2_string = (
            rf".*_cbo_unknown_dr2_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        dr2_results = load_results(base_path, dr2_string)
        dr2_results = np.vstack([dr2_result["Best_Y"] for dr2_result in dr2_results])
        dr2_best_results = dr2_results.min(axis=1)
        dr2_mean = np.mean(dr2_best_results)
        dr2_std = np.std(dr2_best_results)
        min_results["dr2"] = {"mean": dr2_mean, "std": dr2_std}

    if has_cbo:
        cbo_string = rf".*_cbo_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.vstack([cbo_result["Best_Y"] for cbo_result in cbo_results])
        cbo_best_results = cbo_results.min(axis=1)
        cbo_mean = np.mean(cbo_best_results)
        cbo_std = np.std(cbo_best_results)
        min_results["cbo"] = {"mean": cbo_mean, "std": cbo_std}

    if has_random:
        random_string = (
            rf".*_cbo_results_random_{n_obs}_{n_int}{nonlinear_string}.pickle"
        )
        random_results = load_results(base_path, random_string)
        random_results = np.vstack(
            [random_result["Best_Y"] for random_result in random_results]
        )
        random_best_results = random_results.min(axis=1)
        random_mean = np.mean(random_best_results)
        random_std = np.std(random_best_results)
        min_results["random"] = {"mean": random_mean, "std": random_std}

    if has_random:
        random_string = rf".*_cbo_results_{n_obs}_{n_int}{nonlinear_string}.pickle"
        cbo_results = load_results(base_path, cbo_string)
        cbo_results = np.vstack([cbo_result["Best_Y"] for cbo_result in cbo_results])
        cbo_best_results = cbo_results.min(axis=1)
        cbo_mean = np.mean(cbo_best_results)
        cbo_std = np.std(cbo_best_results)
        min_results["cbo"] = {"mean": cbo_mean, "std": cbo_std}

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
