import os
import warnings
from typing import List, Optional

import matplotlib.pyplot as plt
import numpy as np

N_SAMPLES_PLOT = 6
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]


# Plotting utils
def plot_trajs_univariate(
    dyst_data: np.ndarray,
    selected_dim: Optional[int] = None,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    samples_subset: Optional[List[int]] = None,
    n_samples_plot: Optional[int] = None,
) -> None:
    """
    Plot univariate timeseries from dyst_data
    """
    os.makedirs(save_dir, exist_ok=True)
    if n_samples_plot is None:
        num_tot_samples = dyst_data.shape[0]
        n_samples_plot = min(N_SAMPLES_PLOT, num_tot_samples)
    else:
        n_samples_plot = min(N_SAMPLES_PLOT, n_samples_plot)

    if samples_subset is not None:
        if n_samples_plot > len(samples_subset):
            warnings.warn(
                f"Number of samples to plot is greater than the number of samples in the subset. Plotting all {len(samples_subset)} samples in the subset."
            )
            n_samples_plot = len(samples_subset)

    if selected_dim is None:
        dims = list(range(dyst_data.shape[1]))
    else:
        dims = [selected_dim]

    for dim_idx in dims:
        plot_name_dim = f"{plot_name}_dim{dim_idx}"
        save_path = os.path.join(save_dir, f"{plot_name_dim}.png")
        print(
            f"Plotting 2D trajectories for {plot_name}, dimension {dim_idx} and saving to {save_path}"
        )
        plt.figure(figsize=(6, 6))
        for sample_idx in range(n_samples_plot):
            label_sample_idx = sample_idx
            if samples_subset is not None:
                label_sample_idx = samples_subset[sample_idx]
            curr_color = COLORS[label_sample_idx]
            plt.plot(
                dyst_data[sample_idx, dim_idx, :],
                alpha=0.5,
                linewidth=1,
                color=curr_color,
                label=f"Sample {label_sample_idx}",
            )
        plt.xlabel("timesteps")
        plt.title(plot_name_dim.replace("_", " "))
        plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_trajs_multivariate(
    dyst_data: np.ndarray,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    samples_subset: Optional[List[int]] = None,
    n_samples_plot: Optional[int] = None,
    plot_2d_slice: bool = True,
) -> None:
    """
    Plot multivariate timeseries from dyst_data

    Args:
        dyst_data (np.ndarray): Array of shape (n_samples, n_dimensions, n_timesteps) containing the multivariate time series data.
        save_dir (str, optional): Directory to save the plots. Defaults to "tests/figs".
        plot_name (str, optional): Base name for the saved plot files. Defaults to "dyst".
        samples_subset (List[int], optional): Subset of sample indices to plot. If None, all samples are used. Defaults to None.
        n_samples_plot (int, optional): Number of samples to plot. If None, all samples are plotted. Defaults to None.
        plot_2d_slice (bool, optional): Whether to plot a 2D slice of the first two dimensions. Defaults to True.
    """
    os.makedirs(save_dir, exist_ok=True)

    n_samples_plot = dyst_data.shape[0] if n_samples_plot is None else n_samples_plot

    if samples_subset is not None:
        if n_samples_plot > len(samples_subset):
            warnings.warn(
                f"Number of samples to plot is greater than the number of samples in the subset. Plotting all {len(samples_subset)} samples in the subset."
            )
            n_samples_plot = len(samples_subset)

    if plot_2d_slice:
        save_path = os.path.join(save_dir, f"{plot_name}.png")
        print("Plotting 2D trajectories and saving to ", save_path)
        plt.figure(figsize=(6, 6))
        for sample_idx in range(n_samples_plot):
            label_sample_idx = (
                samples_subset[sample_idx] if samples_subset is not None else sample_idx
            )
            label = f"Sample {label_sample_idx}"
            curr_color = COLORS[label_sample_idx]

            xy = dyst_data[sample_idx, :2, :]
            plt.plot(*xy, alpha=0.5, linewidth=1, color=curr_color, label=label)

            ic_point = dyst_data[sample_idx, :2, 0]
            plt.scatter(*ic_point, marker="*", s=100, alpha=0.5, c=curr_color)

            final_point = dyst_data[sample_idx, :2, -1]
            plt.scatter(*final_point, marker="x", s=100, alpha=0.5, c=curr_color)

        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title(plot_name.replace("_", " "))
        plt.savefig(save_path, dpi=300)
        plt.close()

    # 3D plot (first three coordinates)
    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    print("Plotting 3D trajectories and saving to ", save_path)
    if dyst_data.shape[1] >= 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for sample_idx in range(n_samples_plot):
            label_sample_idx = (
                samples_subset[sample_idx] if samples_subset is not None else sample_idx
            )
            label = f"Sample {label_sample_idx}"
            curr_color = COLORS[label_sample_idx]

            xyz = dyst_data[sample_idx, :3, :]
            ax.plot(*xyz, alpha=0.5, linewidth=1, color=curr_color, label=label)

            ic_point = dyst_data[sample_idx, :3, 0]
            ax.scatter(*ic_point, marker="*", s=100, alpha=0.5, c=curr_color)

            final_point = dyst_data[sample_idx, :3, -1]
            ax.scatter(*final_point, marker="x", s=100, alpha=0.5, c=curr_color)

        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        plt.legend()
        ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
        plt.title(plot_name.replace("_", " "))
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_forecast_trajs_multivariate(
    dyst_data: np.ndarray,
    context_length: int,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    n_samples_plot: Optional[int] = None,
) -> None:
    """
    Plot multivariate timeseries from dyst_data
    """
    dyst_name = plot_name.split("_")[0]
    print("Plotting forecast vs ground truth for ", dyst_name)
    os.makedirs(save_dir, exist_ok=True)
    if n_samples_plot is None:
        num_tot_samples = dyst_data.shape[0]
        n_samples_plot = min(N_SAMPLES_PLOT, num_tot_samples)
    else:
        n_samples_plot = min(N_SAMPLES_PLOT, n_samples_plot)

    # Plot the first two coordinates
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    print("Plotting 2D trajectories and saving to ", save_path)
    plt.figure(figsize=(6, 6))
    for sample_idx in range(n_samples_plot):
        curr_color = COLORS[sample_idx]
        plt.scatter(
            *dyst_data[sample_idx, :2, 0],
            marker="*",
            s=25,
            alpha=1,
            color=curr_color,
        )
        # plot x and y
        plt.plot(
            dyst_data[sample_idx, 0, :context_length],
            dyst_data[sample_idx, 1, :context_length],
            alpha=0.25,
            linewidth=1,
            color=curr_color,
        )
        plt.scatter(
            *dyst_data[sample_idx, :2, context_length],
            marker="*",
            s=100,
            alpha=1,
            color=curr_color,
        )
        # plot x and y
        plt.plot(
            dyst_data[sample_idx, 0, context_length:],
            dyst_data[sample_idx, 1, context_length:],
            alpha=0.5,
            linewidth=2,
            color=curr_color,
        )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{dyst_name} Forecast")
    plt.savefig(save_path, dpi=300)
    plt.close()

    # 3D plot (first three coordinates)
    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    print("Plotting 3D trajectories and saving to ", save_path)
    if dyst_data.shape[1] >= 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for sample_idx in range(n_samples_plot):
            curr_color = COLORS[sample_idx]
            ax.scatter(
                *dyst_data[sample_idx, :3, 0],
                marker="*",
                s=25,
                alpha=1,
                color=curr_color,
            )
            # plot x and y and z
            ax.plot(
                dyst_data[sample_idx, 0, :context_length],
                dyst_data[sample_idx, 1, :context_length],
                dyst_data[sample_idx, 2, :context_length],
                alpha=0.5,
                linewidth=1,
                color=curr_color,
            )
            ax.scatter(
                *dyst_data[sample_idx, :3, context_length],
                marker="*",
                s=100,
                alpha=1,
                color=curr_color,
            )
            ax.plot(
                dyst_data[sample_idx, 0, context_length:],
                dyst_data[sample_idx, 1, context_length:],
                dyst_data[sample_idx, 2, context_length:],
                alpha=0.5,
                linewidth=2,
                color=curr_color,
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
        plt.title(f"{dyst_name} Forecast")
        plt.savefig(save_path, dpi=300)
        plt.close()


def plot_forecast_gt_trajs_multivariate(
    fc_data: np.ndarray,
    gt_data: np.ndarray,
    context_length: int,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    n_samples_plot: Optional[int] = None,
) -> None:
    """
    Plot multivariate timeseries from ground trugh and forecasted data
    """
    dyst_name = plot_name.split("_")[0]
    print("Plotting forecast vs ground truth for ", dyst_name)
    os.makedirs(save_dir, exist_ok=True)
    if n_samples_plot is None:
        num_tot_samples = gt_data.shape[0]
        assert num_tot_samples == fc_data.shape[0], "Mismatch in number of samples"
        n_samples_plot = min(N_SAMPLES_PLOT, num_tot_samples)
    else:
        n_samples_plot = min(N_SAMPLES_PLOT, n_samples_plot)

    # Plot the first two coordinates
    save_path = os.path.join(save_dir, f"{plot_name}.png")
    print("Plotting 2D trajectories and saving to ", save_path)
    plt.figure(figsize=(6, 6))
    for sample_idx in range(n_samples_plot):
        curr_color = COLORS[sample_idx]
        plt.scatter(
            *gt_data[sample_idx, :2, 0],
            marker="*",
            s=25,
            alpha=1,
            color=curr_color,
        )
        # plot x and y
        plt.plot(
            gt_data[sample_idx, 0, :],
            gt_data[sample_idx, 1, :],
            alpha=0.25,
            linewidth=1,
            color=curr_color,
            label="Ground Truth",
        )
        plt.scatter(
            *gt_data[sample_idx, :2, context_length],
            marker="*",
            s=100,
            alpha=1,
            color=curr_color,
        )
        # plot x and y
        plt.plot(
            fc_data[sample_idx, 0, context_length:],
            fc_data[sample_idx, 1, context_length:],
            alpha=0.5,
            linewidth=1,
            linestyle="dashed",  # Set the linestyle to dashed
            color=curr_color,
            label="Forecast",
        )
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"{dyst_name} Forecast vs Ground Truth")
    plt.legend()
    plt.savefig(save_path, dpi=300)
    plt.close()

    # 3D plot (first three coordinates)
    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    print("Plotting 3D trajectories and saving to ", save_path)
    if gt_data.shape[1] >= 3:
        fig = plt.figure(figsize=(6, 6))
        ax = fig.add_subplot(111, projection="3d")
        for sample_idx in range(n_samples_plot):
            curr_color = COLORS[sample_idx]
            ax.scatter(
                *gt_data[sample_idx, :3, 0],
                marker="*",
                s=25,
                alpha=1,
                color=curr_color,
            )
            # plot x and y and z
            ax.plot(
                gt_data[sample_idx, 0, :],
                gt_data[sample_idx, 1, :],
                gt_data[sample_idx, 2, :],
                alpha=0.5,
                linewidth=1,
                color=curr_color,
                label="Ground Truth",
            )
            ax.scatter(
                *gt_data[sample_idx, :3, context_length],
                marker="*",
                s=100,
                alpha=1,
                color=curr_color,
            )
            ax.plot(
                fc_data[sample_idx, 0, context_length:],
                fc_data[sample_idx, 1, context_length:],
                fc_data[sample_idx, 2, context_length:],
                alpha=0.5,
                linewidth=1,
                linestyle="dashed",  # Set the linestyle to dashed
                color=curr_color,
                label="Forecast",
            )
        ax.set_xlabel("X")
        ax.set_ylabel("Y")
        ax.set_zlabel("Z")  # type: ignore
        ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
        ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
        plt.title(f"{dyst_name} Forecast vs Ground Truth")
        # plt.legend()
        plt.savefig(save_path, dpi=300)
        plt.close()
