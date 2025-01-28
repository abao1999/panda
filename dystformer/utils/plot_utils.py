import os
import warnings

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TABLEAU_COLORS
from matplotlib.ticker import FormatStrFormatter

from dystformer.utils import safe_standardize

COLORS = list(TABLEAU_COLORS.values())

if os.path.exists("custom_style.mplstyle"):
    plt.style.use(["ggplot", "custom_style.mplstyle"])


def plot_trajs_multivariate(
    trajectories: np.ndarray,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    samples_subset: list[int] | None = None,
    plot_2d_slice: bool = False,
    plot_projections: bool = False,
    standardize: bool = False,
    dims_3d: list[int] = [0, 1, 2],
    figsize: tuple[int, int] = (6, 6),
    max_samples: int = 6,
) -> None:
    """
    Plot multivariate timeseries from dyst_data

    Args:
        trajectories (np.ndarray): Array of shape (n_samples, n_dimensions, n_timesteps) containing the multivariate time series data.
        save_dir (str, optional): Directory to save the plots. Defaults to "tests/figs".
        plot_name (str, optional): Base name for the saved plot files. Defaults to "dyst".
        samples_subset (list[int] | None): Subset of sample indices to plot. If None, all samples are used. Defaults to None.
        plot_2d_slice (bool): Whether to plot a 2D slice of the first two dimensions. Defaults to True.
        plot_projections (bool): Whether to plot 2D projections on the coordinate planes
        standardize (bool): Whether to standardize the trajectories
        dims_3d (list[int]): Indices of dimensions to plot in 3D visualization. Defaults to [0, 1, 2]
        figsize (tuple[int, int]): Figure size in inches (width, height). Defaults to (6, 6)
        max_samples (int): Maximum number of samples to plot. Defaults to 6.
    """
    os.makedirs(save_dir, exist_ok=True)
    assert trajectories.shape[1] >= len(dims_3d), (
        f"Data has {trajectories.shape[1]} dimensions, but {len(dims_3d)} dimensions were requested for plotting"
    )

    n_samples_plot = min(max_samples, trajectories.shape[0])

    if samples_subset is not None:
        if n_samples_plot > len(samples_subset):
            warnings.warn(
                f"Number of samples to plot is greater than the number of samples in the subset. Plotting all {len(samples_subset)} samples in the subset."
            )
            n_samples_plot = len(samples_subset)

    if standardize:
        trajectories = safe_standardize(trajectories)

    if plot_2d_slice:
        save_path = os.path.join(save_dir, f"{plot_name}.png")
        plt.figure(figsize=figsize)
        for sample_idx in range(n_samples_plot):
            label_sample_idx = (
                samples_subset[sample_idx] if samples_subset is not None else sample_idx
            )
            label = f"Sample {label_sample_idx}"
            curr_color = COLORS[sample_idx % len(COLORS)]

            xy = trajectories[sample_idx, :2, :]
            plt.plot(*xy, alpha=0.5, linewidth=1, color=curr_color, label=label)

            ic_point = trajectories[sample_idx, :2, 0]
            plt.scatter(*ic_point, marker="*", s=100, alpha=0.5, color=curr_color)

            final_point = trajectories[sample_idx, :2, -1]
            plt.scatter(*final_point, marker="x", s=100, alpha=0.5, color=curr_color)

        print(f"Saving 2D plot to {save_path}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.legend()
        plt.title(plot_name.replace("_", " "))
        plt.savefig(save_path, dpi=300)
        plt.close()

    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    if n_samples_plot == 1:
        linewidth = 1
    else:
        linewidth = 0.5

    for sample_idx in range(n_samples_plot):
        label_sample_idx = (
            samples_subset[sample_idx] if samples_subset is not None else sample_idx
        )
        label = f"Sample {label_sample_idx}"
        curr_color = COLORS[sample_idx % len(COLORS)]

        xyz = trajectories[sample_idx, dims_3d, :]
        ax.plot(*xyz, alpha=0.5, linewidth=linewidth, color=curr_color, label=label)

        ic_pt = xyz[:, 0]
        ax.scatter(*ic_pt, marker="*", s=100, alpha=0.5, color=curr_color)

        end_pt = xyz[:, -1]
        ax.scatter(*end_pt, marker="x", s=100, alpha=0.5, color=curr_color)

    if plot_projections:
        x_min, x_max = ax.get_xlim3d()  # type: ignore
        y_min, y_max = ax.get_ylim3d()  # type: ignore
        z_min, z_max = ax.get_zlim3d()  # type: ignore
        palpha = 0.1  # whatever

        for sample_idx in range(n_samples_plot):
            label_sample_idx = (
                samples_subset[sample_idx] if samples_subset is not None else sample_idx
            )
            curr_color = COLORS[sample_idx % len(COLORS)]
            xyz = trajectories[sample_idx, dims_3d, :]
            ic_pt = xyz[:, 0]
            end_pt = xyz[:, -1]

            # XY plane projection (bottom)
            ax.plot(xyz[0], xyz[1], z_min, alpha=palpha, linewidth=1, color=curr_color)
            ax.scatter(
                ic_pt[0], ic_pt[1], z_min, marker="*", alpha=palpha, color=curr_color
            )
            ax.scatter(
                end_pt[0], end_pt[1], z_min, marker="x", alpha=palpha, color=curr_color
            )

            # XZ plane projection (back)
            ax.plot(xyz[0], y_max, xyz[2], alpha=palpha, linewidth=1, color=curr_color)
            ax.scatter(
                ic_pt[0], y_max, ic_pt[2], marker="*", alpha=palpha, color=curr_color
            )
            ax.scatter(
                end_pt[0], y_max, end_pt[2], marker="x", alpha=palpha, color=curr_color
            )

            # YZ plane projection (right)
            ax.plot(x_min, xyz[1], xyz[2], alpha=palpha, linewidth=1, color=curr_color)
            ax.scatter(
                x_min, ic_pt[1], ic_pt[2], marker="*", alpha=palpha, color=curr_color
            )
            ax.scatter(
                x_min, end_pt[1], end_pt[2], marker="x", alpha=palpha, color=curr_color
            )

    print(f"Saving 3D plot to {save_path}")
    ax.set_xlabel(f"dim_{dims_3d[0]}")
    ax.set_ylabel(f"dim_{dims_3d[1]}")
    ax.set_zlabel(f"dim_{dims_3d[2]}")  # type: ignore
    plt.legend()
    ax.tick_params(pad=3)
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    plt.title(plot_name.replace("_", " "))
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_grid_trajs_multivariate(
    ensemble: dict[str, np.ndarray],
    save_path: str,
    standardize: bool = False,
    dims_3d: list[int] = [0, 1, 2],
    subplot_size: tuple[int, int] = (3, 3),
    max_samples: int = 6,
) -> None:
    """
    Plot a grid of multiple systems' multivariate timeseries from dyst_data

    Args:
        ensemble (dict[str, np.ndarray]): Dictionary of shape (n_samples, n_dimensions, n_timesteps) containing the multivariate time series data.
        save_dir (str, optional): Directory to save the plots. Defaults to "tests/figs".
        standardize (bool): Whether to standardize the trajectories
        dims_3d (list[int]): Indices of dimensions to plot in 3D visualization. Defaults to [0, 1, 2]
        figsize (tuple[int, int]): Figure size in inches (width, height). Defaults to (6, 6)
        max_samples (int): Maximum number of samples to plot. Defaults to 6.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    print("Plotting grid of 3D trajectories and saving to ", save_path)
    n_systems = len(ensemble)
    n_rows = int(np.ceil(np.sqrt(n_systems)))
    n_cols = int(np.ceil(n_systems / n_rows))

    row_padding = 0.2
    column_padding = 0.2
    figsize = (
        n_cols * subplot_size[0] * (1 + column_padding),
        n_rows * subplot_size[1] * (1 + row_padding),
    )
    fig = plt.figure(figsize=figsize)

    for idx, (system_name, trajectories) in enumerate(ensemble.items()):
        assert trajectories.shape[1] >= len(dims_3d), (
            f"Data has {trajectories.shape[1]} dimensions, but {len(dims_3d)} dimensions were requested for plotting"
        )
        n_samples_plot = min(max_samples, trajectories.shape[0])

        if standardize:
            trajectories = safe_standardize(trajectories)

        ax = fig.add_subplot(n_rows, n_cols, idx + 1, projection="3d")

        if n_samples_plot == 1:
            linewidth = 1
        else:
            linewidth = 0.5
        for sample_idx in range(n_samples_plot):
            label = f"Sample {sample_idx}"
            curr_color = COLORS[sample_idx % len(COLORS)]

            xyz = trajectories[sample_idx, dims_3d, :]
            ax.plot(*xyz, alpha=0.5, linewidth=linewidth, color=curr_color, label=label)

            ic_pt = xyz[:, 0]
            ax.scatter(*ic_pt, marker="*", s=100, alpha=0.5, color=curr_color)

            end_pt = xyz[:, -1]
            ax.scatter(*end_pt, marker="x", s=100, alpha=0.5, color=curr_color)

        system_name_title = system_name.replace("_", " + ")
        ax.set_title(f"{system_name_title}.", fontsize=18, fontweight="bold")
        fig.patch.set_facecolor("white")  # Set the figure's face color to white
        ax.set_facecolor("white")  # Set the axes' face color to white
        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
        ax.zaxis.set_major_formatter(FormatStrFormatter("%.1f"))  # type: ignore
        ax.tick_params(pad=1)
        ax.grid(False)

    # Adjust the layout to add more space between rows
    plt.tight_layout()  # Adjust 'pad' as needed
    plt.subplots_adjust(hspace=row_padding, wspace=column_padding)

    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_completions_evaluation(
    completions: np.ndarray,
    context: np.ndarray,
    mask: np.ndarray | None = None,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    samples_subset: list[int] | None = None,
    max_samples: int = 6,
) -> None:
    """
    Plot side-by-side 3D multivariate timeseries for completions and context,
    and overlay univariate series for each dimension below with shared legends.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_tot_samples = min(completions.shape[0], context.shape[0])
    n_samples_plot = min(max_samples, num_tot_samples)

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
        curr_color = COLORS[sample_idx % len(COLORS)]

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
        ax1.set_title("Context", y=0.94, fontweight="bold")
        ax1.axis("off")
        ax1.grid(b=None)
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
        ax2.set_title("Completions", y=0.94, fontweight="bold")
        ax2.axis("off")
        ax2.grid(b=None)

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
            ax.set_title(f"Dimension {dim + 1}", fontweight="bold")
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
    )  # $), y=0.95)

    # plt.subplots_adjust(hspace=0.6)  # Fine-tune spacing between rows
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{plot_name}_combined.pdf")
    plt.savefig(save_path, dpi=300)
    plt.close()


def plot_forecast_evaluation(
    forecasts: np.ndarray,
    ground_truth: np.ndarray,
    context_length: int,
    save_dir: str = "tests/figs",
    plot_name: str = "dyst",
    samples_subset: list[int] | None = None,
    max_samples: int = 6,
) -> None:
    """
    Plot side-by-side 3D multivariate timeseries for completions and context,
    and overlay univariate series for each dimension below with shared legends.
    """
    os.makedirs(save_dir, exist_ok=True)
    num_tot_samples = min(forecasts.shape[0], ground_truth.shape[0])
    n_samples_plot = min(max_samples, num_tot_samples)
    forecast_length = forecasts.shape[2] - context_length
    print(f"Forecast length: {forecast_length}")
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
        curr_color = COLORS[sample_idx % len(COLORS)]

        # Plot context in 3D
        (line1,) = ax1.plot(
            ground_truth[sample_idx, 0, :context_length],
            ground_truth[sample_idx, 1, :context_length],
            ground_truth[sample_idx, 2, :context_length],
            alpha=0.2,
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
        ax1.set_title("Ground Truth", fontweight="bold", y=0.94)
        ax1.grid(b=None)
        ax1.axis("off")

        # Plot forecasts in 3D
        (line2,) = ax2.plot(
            forecasts[sample_idx, 0, :context_length],
            forecasts[sample_idx, 1, :context_length],
            forecasts[sample_idx, 2, :context_length],
            alpha=0.2,
            linewidth=1,
            color=curr_color,
        )
        ax2.plot(
            forecasts[sample_idx, 0, context_length:],
            forecasts[sample_idx, 1, context_length:],
            forecasts[sample_idx, 2, context_length:],
            alpha=0.8,
            linewidth=2,
            color=curr_color,
            label=f"Sample {label_sample_idx} Forecasts",
        )
        ax2.set_title("Forecasts", fontweight="bold", y=0.94)
        ax2.grid(b=None)
        ax2.axis("off")

        # Add lines and labels for the first sample only to avoid duplicates
        if sample_idx == 0:
            lines.append(line1)
            labels.append(f"Sample {label_sample_idx}")

        # Plot univariate series for each dimension
        for dim, ax in enumerate([ax3, ax4, ax5]):
            ax.plot(
                ground_truth[sample_idx, dim, context_length:],
                alpha=0.2,
                linewidth=1,
                color=curr_color,
                linestyle="-",
            )
            ax.plot(
                forecasts[sample_idx, dim, context_length:],
                alpha=1,
                linewidth=1,
                color=curr_color,
                linestyle="--",
            )

            # Fill between completions and context where mask is False
            ax.fill_between(
                range(0, forecast_length),
                ground_truth[sample_idx, dim, context_length:],
                forecasts[sample_idx, dim, context_length:],
                color=curr_color,
                alpha=0.3,
            )
            ax.set_title(f"Dimension {dim + 1}", fontweight="bold")
            # ax.set_xlabel("Timesteps")
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

    plt.suptitle(plot_name.replace("_", " + "), fontsize=18, fontweight="bold")
    save_path = os.path.join(save_dir, f"{plot_name}_combined.png")
    plt.savefig(save_path, dpi=300)
    plt.close()
