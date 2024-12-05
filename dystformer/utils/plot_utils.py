import os
import warnings
from typing import List, Optional

import matplotlib.cm as cm
import matplotlib.colors as mcolors
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np

N_SAMPLES_PLOT = 6
plt.style.use(["ggplot", "custom_style.mplstyle"])
colormap = cm.get_cmap("tab10", 10)  # 'tab10' is a colormap with 10 distinct colors
COLORS = [mcolors.rgb2hex(colormap(i)) for i in range(colormap.N)]


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

    if plot_2d_slice:
        save_path = os.path.join(save_dir, f"{plot_name}.png")
        print("Plotting 2D trajectories and saving to ", save_path)
        plt.figure(figsize=(6, 6))
        for sample_idx in range(n_samples_plot):
            label_sample_idx = (
                samples_subset[sample_idx] if samples_subset is not None else sample_idx
            )
            label = f"Sample {label_sample_idx}"
            print(f"Plotting sample {label_sample_idx}")
            curr_color = COLORS[label_sample_idx % len(COLORS)]

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
            curr_color = COLORS[label_sample_idx % len(COLORS)]

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


def plot_completions_evaluation(
    completions: np.ndarray,
    context: np.ndarray,
    mask: Optional[np.ndarray] = None,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    samples_subset: Optional[List[int]] = None,
) -> None:
    """
    Plot side-by-side 3D multivariate timeseries for completions and context,
    and overlay univariate series for each dimension below with shared legends.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_tot_samples = min(completions.shape[0], context.shape[0])
    n_samples_plot = min(N_SAMPLES_PLOT, num_tot_samples)

    if samples_subset is not None:
        if n_samples_plot > len(samples_subset):
            warnings.warn(
                f"Number of samples to plot is greater than the number of samples in the subset. Plotting all {len(samples_subset)} samples in the subset."
            )
            n_samples_plot = len(samples_subset)

    # Create a figure with 4 rows: one for 3D plots and three for univariate plots
    fig = plt.figure(figsize=(12, 15))  # Adjusted height for more rows

    # Create a GridSpec for the layout (make first row twice as tall)
    gs = gridspec.GridSpec(4, 2, height_ratios=[2, 1, 1, 1])
    # Top row: ax1 and ax2 share a row
    ax1 = fig.add_subplot(gs[0, 0], projection="3d")  # Top-left
    ax2 = fig.add_subplot(gs[0, 1], projection="3d")  # Top-right

    # ax3, ax4, ax5 occupy their own rows
    ax3 = fig.add_subplot(gs[1, :])  # Second row, spans both columns
    ax4 = fig.add_subplot(gs[2, :])  # Third row, spans both columns
    ax5 = fig.add_subplot(gs[3, :])  # Fourth row, spans both columns

    # Collect lines and labels for a shared legend
    lines = []
    labels = []

    # Plot 3D trajectories
    for sample_idx in range(n_samples_plot):
        label_sample_idx = (
            samples_subset[sample_idx] if samples_subset is not None else sample_idx
        )
        curr_color = COLORS[label_sample_idx % len(COLORS)]

        # Plot context in 3D
        (line1,) = ax1.plot(
            context[sample_idx, 0, :],
            context[sample_idx, 1, :],
            context[sample_idx, 2, :],
            alpha=0.5,
            linewidth=1,
            color=curr_color,
            label=f"Sample {label_sample_idx}",
        )
        ax1.set_title("Context")
        # Plot completions in 3D
        (line2,) = ax2.plot(
            completions[sample_idx, 0, :],
            completions[sample_idx, 1, :],
            completions[sample_idx, 2, :],
            alpha=0.5,
            linewidth=1,
            color=curr_color,
            label=f"Sample {label_sample_idx}",
        )
        ax2.set_title("Completions")

        # Add lines and labels for the first sample only to avoid duplicates
        if sample_idx == 0:
            lines.append(line1)
            labels.append(f"Sample {label_sample_idx}")

        # Plot univariate series for each dimension
        for dim, ax in enumerate([ax3, ax4, ax5]):
            # Ensure mask is a boolean array
            if mask is not None:
                mask_bool = mask[sample_idx, dim, :].astype(bool)
            else:
                mask_bool = np.ones(
                    context.shape[2], dtype=bool
                )  # Default to all True if no mask

            # Find the indices where the mask changes
            diffs = np.diff(mask_bool.astype(int))
            change_indices = np.where(diffs)[0]
            if not mask_bool[0]:
                change_indices = np.concatenate(([0], change_indices))
            segment_indices = np.concatenate((change_indices, [context.shape[2]]))
            ax.plot(
                context[sample_idx, dim, :],
                alpha=0.2,
                linewidth=1,
                color=curr_color,
                linestyle="-",
            )
            # Plot context and completions with varying line widths
            segments = zip(segment_indices[:-1], segment_indices[1:])
            masked_segments = [
                idx for i, idx in enumerate(segments) if (i + 1) % 2 == 1
            ]
            for start, end in masked_segments:
                if end < completions.shape[2] - 1:
                    end += 1
                ax.plot(
                    range(start, end),  # Ensure end is inclusive
                    completions[sample_idx, dim, start:end],
                    alpha=1,
                    linewidth=1,
                    color=curr_color,
                    linestyle="--",
                )

            # Fill between completions and context where mask is False
            ax.fill_between(
                range(context.shape[2]),  # Assuming the mask is 1D over timesteps
                context[sample_idx, dim, :],
                completions[sample_idx, dim, :],
                where=~mask_bool,  # type: ignore
                color=curr_color,
                alpha=0.3,
            )
            ax.set_title(f"Dimension {dim + 1}")
            # ax.set_xlabel("Timesteps")

    # # Create a shared legend for samples
    # fig.legend(lines, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))

    # Add a legend for line styles in the univariate plots
    line_context = plt.Line2D([0], [0], color="black", linestyle="-", label="Context")  # type: ignore
    line_completions = plt.Line2D(  # type: ignore
        [0], [0], color="black", linestyle="--", label="Completions"
    )

    ax3.legend(handles=[line_context, line_completions], loc="upper right")

    plt.suptitle(
        plot_name.replace("_", " + "), fontsize=18, fontweight="bold"
    )  # Adjust y for padding
    # plt.subplots_adjust(hspace=0.6)  # Fine-tune spacing between rows
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{plot_name}_combined.png")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_forecast_evaluation(
    forecasts: np.ndarray,
    ground_truth: np.ndarray,
    context_length: int,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    samples_subset: Optional[List[int]] = None,
) -> None:
    """
    Plot side-by-side 3D multivariate timeseries for completions and context,
    and overlay univariate series for each dimension below with shared legends.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_tot_samples = min(forecasts.shape[0], ground_truth.shape[0])
    n_samples_plot = min(N_SAMPLES_PLOT, num_tot_samples)

    if samples_subset is not None:
        if n_samples_plot > len(samples_subset):
            warnings.warn(
                f"Number of samples to plot is greater than the number of samples in the subset. Plotting all {len(samples_subset)} samples in the subset."
            )
            n_samples_plot = len(samples_subset)

    # Create a figure with 2 rows: one for 3D plots and one for univariate plots
    fig = plt.figure(figsize=(16, 12))
    ax1 = fig.add_subplot(221, projection="3d")
    ax2 = fig.add_subplot(222, projection="3d")
    ax3 = fig.add_subplot(234)
    ax4 = fig.add_subplot(235)
    ax5 = fig.add_subplot(236)

    # Collect lines and labels for a shared legend
    lines = []
    labels = []

    # Plot 3D trajectories
    for sample_idx in range(n_samples_plot):
        label_sample_idx = (
            samples_subset[sample_idx] if samples_subset is not None else sample_idx
        )
        curr_color = COLORS[label_sample_idx % len(COLORS)]

        # Plot context in 3D
        (line1,) = ax1.plot(
            ground_truth[sample_idx, 0, :context_length],
            ground_truth[sample_idx, 1, :context_length],
            ground_truth[sample_idx, 2, :context_length],
            alpha=0.4,
            linewidth=1,  # Thin line for context_length points
            color=curr_color,
        )
        ax1.plot(
            ground_truth[sample_idx, 0, context_length:],
            ground_truth[sample_idx, 1, context_length:],
            ground_truth[sample_idx, 2, context_length:],
            alpha=0.8,
            linewidth=2,  # Thick line for points after context_length
            color=curr_color,
            label=f"Sample {label_sample_idx} Ground Truth",
        )
        ax1.set_title("Ground Truth")
        # Plot forecasts in 3D
        (line2,) = ax2.plot(
            forecasts[sample_idx, 0, :context_length],
            forecasts[sample_idx, 1, :context_length],
            forecasts[sample_idx, 2, :context_length],
            alpha=0.4,
            linewidth=1,
            color=curr_color,
        )
        ax2.set_title("Forecasts")
        ax2.plot(
            forecasts[sample_idx, 0, context_length:],
            forecasts[sample_idx, 1, context_length:],
            forecasts[sample_idx, 2, context_length:],
            alpha=0.8,
            linewidth=2,
            color=curr_color,
            label=f"Sample {label_sample_idx} Forecasts",
        )

        # Add lines and labels for the first sample only to avoid duplicates
        if sample_idx == 0:
            lines.append(line1)
            labels.append(f"Sample {label_sample_idx}")

        # Plot univariate series for each dimension
        for dim, ax in enumerate([ax3, ax4, ax5]):
            ax.plot(
                ground_truth[sample_idx, dim, context_length:],
                alpha=0.5,
                linewidth=1,
                color=curr_color,
                linestyle="-",
            )
            ax.plot(
                forecasts[sample_idx, dim, context_length:],
                alpha=0.5,
                linewidth=1,
                color=curr_color,
                linestyle="--",
            )
            # ax.axvline(x=context_length, color="black", linestyle="-", linewidth=1)
            ax.set_title(f"Dimension {dim + 1}")
            ax.set_xlabel("Timesteps")

    # # Create a shared legend for samples
    # fig.legend(lines, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.95))

    # Add a legend for line styles in the univariate plots
    line_ground_truth = plt.Line2D(  # type: ignore
        [0], [0], color="black", linestyle="-", label="Ground Truth"
    )
    line_forecasts = plt.Line2D(  # type: ignore
        [0], [0], color="black", linestyle="--", label="Forecasts"
    )

    ax3.legend(handles=[line_ground_truth, line_forecasts], loc="upper right")

    plt.suptitle(
        plot_name.replace("_", " + "), fontsize=16, y=1.02
    )  # Adjust y for padding
    save_path = os.path.join(save_dir, f"{plot_name}_combined.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
