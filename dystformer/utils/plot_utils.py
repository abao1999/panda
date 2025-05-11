import os
import warnings
from typing import Any, Literal

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib import patches as mpatches
from matplotlib.patches import FancyArrowPatch
from matplotlib.ticker import FormatStrFormatter
from mpl_toolkits.mplot3d.proj3d import proj_transform
from omegaconf import OmegaConf

from dystformer.utils import safe_standardize

DEFAULT_COLORS = list(plt.rcParams["axes.prop_cycle"].by_key()["color"])
DEFAULT_MARKERS = ["o", "s", "v", "D", "X", "P", "H", "h", "d", "p", "x"]


def apply_custom_style(config_path: str):
    """
    Apply custom matplotlib style from config file with rcparams
    """
    if os.path.exists(config_path):
        cfg = OmegaConf.load(config_path)
        plt.style.use(cfg.base_style)

        custom_rcparams = OmegaConf.to_container(cfg.matplotlib_style, resolve=True)
        for category, settings in custom_rcparams.items():
            if isinstance(settings, dict):
                for param, value in settings.items():
                    if isinstance(value, dict):
                        for subparam, subvalue in value.items():
                            plt.rcParams[f"{category}.{param}.{subparam}"] = subvalue
                    else:
                        plt.rcParams[f"{category}.{param}"] = value
    else:
        print(f"Warning: Plotting config not found at {config_path}")


class Arrow3D(FancyArrowPatch):
    def __init__(self, xs, ys, zs, *args, **kwargs):
        super().__init__((0, 0), (0, 0), *args, **kwargs)
        self._verts3d = xs, ys, zs

    def draw(self, renderer):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, _ = proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        super().draw(renderer)

    def do_3d_projection(self):
        xs3d, ys3d, zs3d = self._verts3d
        xs, ys, zs = proj_transform(xs3d, ys3d, zs3d, self.axes.get_proj())
        self.set_positions((xs[0], ys[0]), (xs[1], ys[1]))
        return np.min(zs)


def make_clean_projection(ax_3d):
    ax_3d.grid(False)
    ax_3d.set_facecolor("white")
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])
    ax_3d.axis("off")


def make_arrow_axes(ax_3d):
    ax_3d.grid(False)
    ax_3d.set_facecolor("white")
    ax_3d.set_xticks([])
    ax_3d.set_yticks([])
    ax_3d.set_zticks([])
    ax_3d.axis("off")

    # Get axis limits
    x0, x1 = ax_3d.get_xlim3d()
    y0, y1 = ax_3d.get_ylim3d()
    z0, z1 = ax_3d.get_zlim3d()

    ax_3d.set_box_aspect((x1 - x0, y1 - y0, z1 - z0))
    # Define arrows along the three frame edges
    edges = [
        ((x0, y0, z0), (x1, y0, z0), "X"),
        ((x0, y0, z0), (x0, y1, z0), "Y"),
        ((x0, y0, z0), (x0, y0, z1), "Z"),
    ]

    for (xs, ys, zs), (xe, ye, ze), label in edges:
        arr = Arrow3D(
            [xs, xe],
            [ys, ye],
            [zs, ze],
            mutation_scale=20,
            lw=1.5,
            arrowstyle="-|>",
            color="black",
        )
        ax_3d.add_artist(arr)
        ax_3d.text(xe * 1.03, ye * 1.03, ze * 1.03, label, fontsize=12)

    # Hide the default frame and ticks
    for pane in (ax_3d.xaxis.pane, ax_3d.yaxis.pane, ax_3d.zaxis.pane):
        pane.set_visible(False)
    ax_3d.view_init(elev=30, azim=30)


def plot_model_prediction(
    pred: np.ndarray,
    context: np.ndarray,
    groundtruth: np.ndarray,
    title: str | None = None,
    save_path: str | None = None,
    show_plot: bool = True,
    use_arrow_axes: bool = False,
    figsize: tuple[int, int] = (6, 8),
):
    prediction_length = pred.shape[1]
    total_length = context.shape[1] + prediction_length
    context_ts = np.arange(context.shape[1]) / total_length
    pred_ts = np.arange(context.shape[1], total_length) / total_length

    # Create figure with gridspec layout
    fig = plt.figure(figsize=figsize)

    # Create main grid with padding for colorbar
    outer_grid = fig.add_gridspec(2, 1, height_ratios=[0.65, 0.35], hspace=-0.1)

    # Create sub-grid for the plots
    gs = outer_grid[1].subgridspec(3, 1, height_ratios=[0.2] * 3, wspace=0, hspace=0)
    ax_3d = fig.add_subplot(outer_grid[0], projection="3d")

    ax_3d.plot(*context[:3], alpha=0.5, color="black", label="Context")
    ax_3d.plot(*groundtruth[:3], linestyle="-", color="black", label="Groundtruth")
    ax_3d.plot(*pred[:3], color="red", label="Prediction")
    if use_arrow_axes:
        make_arrow_axes(ax_3d)
    else:
        make_clean_projection(ax_3d)

    if title is not None:
        title_name = title.replace("_", " ")
        ax_3d.set_title(title_name, fontweight="bold")

    axes_1d = [fig.add_subplot(gs[i, 0]) for i in range(3)]
    for i, ax in enumerate(axes_1d):
        ax.plot(context_ts, context[i], alpha=0.5, color="black")
        ax.plot(pred_ts, groundtruth[i], linestyle="-", color="black")
        ax.plot(pred_ts, pred[:, i], color="red")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_aspect("auto")

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        print(f"saving fig to: {save_path}")
        plt.savefig(save_path, bbox_inches="tight")
    if show_plot:
        plt.show()
    plt.close()


def plot_forecast_3d(
    forecast: np.ndarray,
    ground_truth: np.ndarray,
    context: np.ndarray,
    figsize: tuple[int, int] = (6, 6),
    indices: list[int] = [0, 1, 2],
    save_path: str | None = None,
) -> None:
    """
    Plot a 3D forecast

    Args:
        forecast (np.ndarray): A (num_features, num_timesteps) numpy array containing the forecast
        ground_truth (np.ndarray): A (num_features, num_timesteps) numpy array containing the ground truth
        context (np.ndarray): A (num_features, num_timesteps) numpy array containing the context
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")
    ax.plot(
        context[indices[0]],
        context[indices[1]],
        context[indices[2]],
        color="black",
        label="context",
    )
    ax.plot(
        forecast[indices[0]],
        forecast[indices[1]],
        forecast[indices[2]],
        color="red",
        label="forecast",
    )
    ax.plot(
        ground_truth[indices[0]],
        ground_truth[indices[1]],
        ground_truth[indices[2]],
        color="black",
        linestyle="--",
        label="ground truth",
    )
    ax.legend()
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_forecast_1d(
    forecast: np.ndarray,
    ground_truth: np.ndarray,
    context: np.ndarray,
    figsize: tuple[int, int] = (6, 6),
    save_path: str | None = None,
    indices: list[int] = [0, 1, 2],
    channel_axis: int = 0,
) -> None:
    """
    Plot a 1D forecast

    Args:
        forecast (np.ndarray): A (num_features, num_timesteps) numpy array containing the forecast
        ground_truth (np.ndarray): A (num_features, num_timesteps) numpy array containing the ground truth
        context (np.ndarray): A (num_features, num_timesteps) numpy array containing the context
        save_path (str, optional): Path to save the plot. Defaults to None.
    """
    assert forecast.shape[0] == ground_truth.shape[0] == context.shape[0]
    assert all(0 <= idx < forecast.shape[0] for idx in indices)

    context_ts = np.arange(context.shape[1])
    forecast_ts = np.arange(context.shape[1], context.shape[1] + forecast.shape[1])

    fig, axes = plt.subplots(nrows=3, ncols=1, figsize=figsize, sharex=True)
    for i in range(forecast.shape[0]):
        axes[i].plot(context_ts, context[i], color="black", label="context")
        axes[i].plot(forecast_ts, forecast[i], color="red", label="forecast")
        axes[i].plot(
            forecast_ts,
            ground_truth[i],
            color="black",
            linestyle="--",
            label="ground truth",
        )
    axes[0].legend()

    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    else:
        plt.show()
    plt.close()


def plot_trajs_multivariate(
    trajectories: np.ndarray,
    save_dir: str | None = None,
    plot_name: str = "dyst",
    samples_subset: list[int] | None = None,
    plot_projections: bool = False,
    standardize: bool = False,
    dims_3d: list[int] = [0, 1, 2],
    figsize: tuple[int, int] = (6, 6),
    max_samples: int = 6,
    show_plot: bool = False,
) -> None:
    """
    Plot multivariate timeseries from dyst_data

    Args:
        trajectories (np.ndarray): Array of shape (n_samples, n_dimensions, n_timesteps) containing the multivariate time series data.
        save_dir (str, optional): Directory to save the plots. Defaults to None.
        plot_name (str, optional): Base name for the saved plot files. Defaults to "dyst".
        samples_subset (list[int] | None): Subset of sample indices to plot. If None, all samples are used. Defaults to None.
        plot_projections (bool): Whether to plot 2D projections on the coordinate planes
        standardize (bool): Whether to standardize the trajectories
        dims_3d (list[int]): Indices of dimensions to plot in 3D visualization. Defaults to [0, 1, 2]
        figsize (tuple[int, int]): Figure size in inches (width, height). Defaults to (6, 6)
        max_samples (int): Maximum number of samples to plot. Defaults to 6.
    """
    if save_dir is not None:
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
        curr_color = DEFAULT_COLORS[sample_idx % len(DEFAULT_COLORS)]

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
            curr_color = DEFAULT_COLORS[sample_idx % len(DEFAULT_COLORS)]
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

    save_path = (
        os.path.join(save_dir, f"{plot_name}_3D.png") if save_dir is not None else None
    )
    if save_path is not None:
        print(f"Saving 3D plot to {save_path}")
    ax.set_xlabel(f"dim_{dims_3d[0]}")
    ax.set_ylabel(f"dim_{dims_3d[1]}")
    ax.set_zlabel(f"dim_{dims_3d[2]}")  # type: ignore
    plt.legend()
    ax.tick_params(pad=3)
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    plt.title(plot_name.replace("_", " "))
    if save_path is not None:
        plt.savefig(save_path, dpi=300)
    if show_plot:
        print("Showing plot")
        plt.show()
    plt.close()


def plot_univariate_trajs(
    ensemble: dict[str, np.ndarray],
    save_path: str,
    figsize: tuple[int, int] = (6, 6),
    max_samples: int = 6,
    standardize: bool = False,
    ndims: int | None = None,
) -> None:
    """
    Plot univariate coordinates of multivariate timeseries

    Args:
        ensemble (dict[str, np.ndarray]): Dictionary of shape (n_samples, n_dimensions, n_timesteps) containing the multivariate time series data.
        save_path (str): Path to save the plots.
        standardize (bool): Whether to standardize the trajectories
        figsize (tuple[int, int]): Figure size in inches (width, height). Defaults to (6, 6)
        max_samples (int): Maximum number of samples to plot. Defaults to 6.
    """
    os.makedirs(save_path, exist_ok=True)

    for system_name, trajectories in ensemble.items():
        if standardize:
            trajectories = safe_standardize(trajectories)

        if ndims is None:
            ndims = trajectories.shape[1]

        fig, axes = plt.subplots(nrows=ndims, ncols=1, figsize=figsize, sharex=True)
        for i in range(ndims):
            axes[i].plot(trajectories[:max_samples, i, :].T)
        fig.savefig(os.path.join(save_path, f"{system_name}_coords.png"))
        plt.close()


def plot_grid_trajs_multivariate(
    ensemble: dict[str, np.ndarray],
    save_path: str,
    standardize: bool = False,
    dims_3d: list[int] = [0, 1, 2],
    subplot_size: tuple[int, int] = (3, 3),
    max_samples: int = 6,
    show_plot: bool = False,
) -> None:
    """
    Plot a grid of multiple systems' multivariate timeseries from dyst_data

    Args:
        ensemble (dict[str, np.ndarray]): Dictionary of shape (n_samples, n_dimensions, n_timesteps) containing the multivariate time series data.
        save_dir (str, optional): Directory to save the plots. Defaults to None.
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
            curr_color = DEFAULT_COLORS[sample_idx % len(DEFAULT_COLORS)]

            xyz = trajectories[sample_idx, dims_3d, :]
            ax.plot(*xyz, alpha=0.5, linewidth=linewidth, color=curr_color, label=label)

            ic_pt = xyz[:, 0]
            ax.scatter(*ic_pt, marker="*", s=100, alpha=0.5, color=curr_color)

            end_pt = xyz[:, -1]
            ax.scatter(*end_pt, marker="x", s=100, alpha=0.5, color=curr_color)

        system_name_title = system_name.replace("_", " + ")
        ax.set_title(f"{system_name_title}.", fontweight="bold")
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
    if show_plot:
        plt.show()
    plt.close()


def plot_completions_evaluation(
    completions: np.ndarray,
    context: np.ndarray,
    mask: np.ndarray | None = None,
    save_dir: str | None = None,
    plot_name: str = "dyst",
    samples_subset: list[int] | None = None,
    max_samples: int = 6,
) -> None:
    """
    Plot side-by-side 3D multivariate timeseries for completions and context,
    and overlay univariate series for each dimension below with shared legends.
    """
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
        curr_color = DEFAULT_COLORS[sample_idx % len(DEFAULT_COLORS)]

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

    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{plot_name}_combined.pdf")
        plt.savefig(save_path, dpi=300)
    plt.close()


def plot_forecast_evaluation(
    forecasts: np.ndarray,
    ground_truth: np.ndarray,
    context_length: int,
    save_dir: str | None = None,
    plot_name: str = "dyst",
    samples_subset: list[int] | None = None,
    max_samples: int = 6,
    show_plot: bool = False,
) -> None:
    """
    Plot side-by-side 3D multivariate timeseries for completions and context,
    and overlay univariate series for each dimension below with shared legends.
    """
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
        curr_color = DEFAULT_COLORS[sample_idx % len(DEFAULT_COLORS)]

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
    if save_dir is not None:
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{plot_name}_combined.png")
        plt.savefig(save_path, dpi=300)
    if show_plot:
        plt.show()
    plt.close()


def make_box_plot(
    unrolled_metrics: dict[str, dict[int, dict[str, list[float]]]],
    prediction_length: int,
    metric_to_plot: str = "smape",
    selected_run_names: list[str] | None = None,
    ylim: tuple[float, float] | None = None,
    verbose: bool = False,
    run_names_to_exclude: list[str] = [],
    use_inv_spearman: bool = False,
    title: str | None = None,
    fig_kwargs: dict[str, Any] = {},
    title_kwargs: dict[str, Any] = {},
    colors: list[str] = DEFAULT_COLORS,
    sort_runs: bool = False,
    save_path: str | None = None,
    order_by_metric: str | None = None,
    ylabel_fontsize: int = 8,
    show_xlabel: bool = True,
    show_legend: bool = False,
    legend_kwargs: dict[str, Any] = {},
    alpha_val: float = 0.8,
    box_percentile_range: tuple[int, int] = (25, 75),
    whisker_percentile_range: tuple[float, float] = (5, 95),
    box_width: float = 0.6,
    has_nans: dict[str, bool] | None = None,
) -> list[mpatches.Patch] | None:
    if fig_kwargs == {}:
        fig_kwargs = {"figsize": (3, 5)}

    if selected_run_names is None:
        selected_run_names = list(unrolled_metrics.keys())

    run_names = [
        name for name in selected_run_names if name not in run_names_to_exclude
    ]
    has_nans = has_nans or {}

    if len(run_names) == 0:
        print("No run names to plot after exclusions!")
        return

    plt.figure(**fig_kwargs)
    plot_data = []

    ordering_metric_data = {}

    for run_name in run_names:
        try:
            if prediction_length not in unrolled_metrics[run_name]:
                warnings.warn(
                    f"Warning: prediction_length {prediction_length} not found for {run_name}"
                )
                continue

            # Check if this run has the specified metric
            if metric_to_plot not in unrolled_metrics[run_name][prediction_length]:
                warnings.warn(
                    f"Warning: metric '{metric_to_plot}' not found for {run_name}"
                )
                continue

            values = unrolled_metrics[run_name][prediction_length][metric_to_plot]
            has_nans[run_name] = bool(np.isnan(values).any()) or has_nans.get(
                run_name, False
            )

            # Process values based on metric type
            if metric_to_plot == "spearman" and use_inv_spearman:
                values = [1 - x for x in values]

            # Filter out NaN values
            values = [v for v in values if not np.isnan(v)]

            if len(values) == 0:
                warnings.warn(f"Warning: All values for {run_name} are NaN")
                continue

            median_value = np.median(values)
            plot_data.extend([(run_name, v) for v in values])

            # If we need to order by a different metric, collect that data too
            if order_by_metric is not None and order_by_metric != metric_to_plot:
                if order_by_metric in unrolled_metrics[run_name][prediction_length]:
                    order_values = unrolled_metrics[run_name][prediction_length][
                        order_by_metric
                    ]

                    # Apply same processing as we would for the plotting metric
                    if order_by_metric == "spearman" and use_inv_spearman:
                        order_values = [1 - x for x in order_values]

                    # Filter out NaN values
                    order_values = [v for v in order_values if not np.isnan(v)]

                    if order_values:
                        ordering_metric_data[run_name] = np.median(order_values)

            if verbose:
                print(f"{run_name} median {metric_to_plot}: {median_value}")

        except Exception as e:
            warnings.warn(f"Error processing {run_name}: {e}")

    df = pd.DataFrame(plot_data, columns=["Run", "Value"])

    if order_by_metric is not None and ordering_metric_data:
        run_order = [
            run for run, _ in sorted(ordering_metric_data.items(), key=lambda x: x[1])
        ]
        run_order = [run for run in run_order if run in df["Run"].unique()]
        df["Run"] = pd.Categorical(df["Run"], categories=run_order, ordered=True)
    elif sort_runs:
        # Use the existing sort_runs logic if order_by_metric isn't specified
        median_by_run = df.groupby("Run")["Value"].median().sort_values()
        run_order = median_by_run.index.tolist()
        df["Run"] = pd.Categorical(df["Run"], categories=run_order, ordered=True)

    metric_title = metric_to_plot
    if metric_to_plot in ["mse", "mae", "rmse", "mape"]:
        metric_title = metric_to_plot.upper()
    elif metric_to_plot == "smape":
        metric_title = "sMAPE"
    elif metric_to_plot == "spearman":
        metric_title = "1 - Spearman" if use_inv_spearman else "Spearman"
    else:
        metric_title = metric_to_plot.capitalize()

    ax = plt.gca()
    unique_runs = (
        df["Run"].unique()
        if not isinstance(df["Run"].dtype, pd.CategoricalDtype)
        else df["Run"].cat.categories
    )

    for i, run in enumerate(unique_runs):
        run_data = df[df["Run"] == run]["Value"].to_numpy()
        if len(run_data) == 0:
            continue

        # Calculate the percentiles
        lower_box, upper_box = np.percentile(run_data, box_percentile_range)
        lower_whisker, upper_whisker = np.percentile(run_data, whisker_percentile_range)
        median_val = np.median(run_data)
        if isinstance(colors, dict):
            color = colors[run]
        else:
            color = colors[i % len(colors)]  # type: ignore
        box_half_width = box_width / 2
        whisker_cap_width = box_half_width * 0.5

        box = plt.Rectangle(
            (i - box_half_width, lower_box),
            box_width,
            upper_box - lower_box,
            fill=True,
            facecolor=color,
            alpha=alpha_val,
            linewidth=1,
            edgecolor="black",
            zorder=5,
        )
        ax.add_patch(box)

        ax.hlines(
            median_val,
            i - box_half_width,
            i + box_half_width,
            colors="black",
            linewidth=2.5,
            zorder=10,
        )

        ax.vlines(
            i,
            lower_box,
            lower_whisker,
            colors="black",
            linestyle="-",
            linewidth=1,
            zorder=5,
        )
        ax.vlines(
            i,
            upper_box,
            upper_whisker,
            colors="black",
            linestyle="-",
            linewidth=1,
            zorder=5,
        )

        ax.hlines(
            lower_whisker,
            i - whisker_cap_width,
            i + whisker_cap_width,
            colors="black",
            linewidth=1,
            zorder=5,
        )
        ax.hlines(
            upper_whisker,
            i - whisker_cap_width,
            i + whisker_cap_width,
            colors="black",
            linewidth=1,
            zorder=5,
        )

    if ylim:
        plt.ylim(ylim)

    plt.ylabel(metric_title, fontweight="bold", fontsize=ylabel_fontsize)
    plt.xlabel("")
    if show_xlabel:
        plt.xticks(
            range(len(unique_runs)),
            unique_runs.tolist(),
            rotation=45,
            ha="right",
            fontsize=5,
            fontweight="bold",
        )
    else:
        plt.xticks([])

    if title is not None:
        title_with_metric = f"{title}: {metric_title}" if title == "Metrics" else title
        plt.title(title_with_metric, fontweight="bold", **title_kwargs)

    plt.tight_layout()
    if isinstance(df["Run"].dtype, pd.CategoricalDtype):
        runs = df["Run"].cat.categories.tolist()
    else:
        runs = df["Run"].unique().tolist()

    # Add a dagger to the run name if it has NaNs
    runs = [f"{run}$^\dagger$" if has_nans[run] else run for run in runs]

    if isinstance(colors, dict):
        legend_handles = [
            mpatches.Patch(color=colors[run], label=run, alpha=alpha_val)  # type: ignore
            for run in runs
        ]
    else:
        legend_handles = [
            mpatches.Patch(color=colors[i % len(colors)], label=run, alpha=alpha_val)  # type: ignore
            for i, run in enumerate(runs)
        ]

    if show_legend:
        plt.legend(handles=legend_handles, **legend_kwargs)

    if save_path is not None:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")

    plt.show()
    return legend_handles


def plot_all_metrics_by_prediction_length(
    all_metrics_dict: dict[str, dict[str, dict[str, list[float]]]],
    metric_names: list[str],
    stat_to_plot: Literal["mean", "median"] = "median",
    metrics_to_show_envelope: list[str] = [],
    percentile_range: tuple[int, int] = (25, 75),
    n_rows: int = 2,
    n_cols: int = 3,
    individual_figsize: tuple[int, int] = (4, 4),
    save_path: str | None = None,
    ylim: tuple[float | None, float | None] = (None, None),
    show_legend: bool = True,
    legend_kwargs: dict = {},
    colors: list[str] | dict[str, str] = DEFAULT_COLORS,
    markers: list[str] = DEFAULT_MARKERS,
    use_inv_spearman: bool = False,
    model_names_to_exclude: list[str] = [],
    has_nans: dict[str, dict[str, bool]] | None = None,
    ignore_nans: bool = False,
) -> list[plt.Line2D]:
    """
    Plot multiple metrics across different prediction lengths for various models.

    Parameters:
    -----------
    all_metrics_dict : dict[str, dict[str, dict[str, list[float]]]]
        A nested dictionary with the following structure:
        - First level keys are metric names (e.g., 'mse', 'mae', 'smape', 'spearman')
        - Second level keys are model names (e.g., 'Our Model', 'Chronos 20M', 'TimesFM 200M')
        - Third level contains metric data with keys like:
          - 'prediction_lengths': list of prediction lengths
          - 'means': mean values for each prediction length
          - 'medians': median values for each prediction length
          - 'stds': standard deviations for each prediction length
          - 'stes': standard error of the mean for each prediction length
          - 'all_vals': list of all values for each prediction length

    metric_names : list[str]
        List of metric names to plot (must be keys in all_metrics_dict)

    model_names_to_exclude : list[str], default=[]
        List of model names to exclude from the plot

    stat_to_plot : Literal["mean", "median"], default="median"
        Statistic to plot

    metrics_to_show_envelope : list[str]
        List of metrics for which to show envelopes
        if stat_to_plot == "mean", then show standard *error* envelopes
        if stat_to_plot == "median", then show median envelopes

    percentile_range : tuple[int, int], default=(25, 75)
        Percentile range to use for the median envelope, only used when stat_to_plot == "median"

    n_rows : int, default=2
        Number of rows in the subplot grid

    n_cols : int, default=3
        Number of columns in the subplot grid

    individual_figsize : tuple[int, int], default=(4, 4)
        Size of each individual subplot

    save_path : str | None, default=None
        Path to save the figure, if None, the figure is not saved

    ylim : tuple[float | None, float | None], default=(None, None)
        Y-axis limits for all subplots

    show_legend : bool, default=True
        Whether to show the legend

    legend_kwargs : dict, default={}
        Additional keyword arguments for the legend

    colors : list[str] | dict[str, str], default=default_colors
        List of colors to use for different models
        Or, dict of model names to colors

    markers : list[str], default=markers
        List of markers to use for different models

    use_inv_spearman : bool, default=False
        If True, plot 1 - Spearman correlation instead of Spearman correlation

    Returns:
    --------
    list[plt.Line2D]
        Legend handles for the plotted lines
    """
    has_nans = has_nans or {}
    num_metrics = len(metric_names)
    fig, axes = plt.subplots(
        nrows=n_rows,
        ncols=n_cols,
        figsize=(individual_figsize[0] * n_cols, individual_figsize[1] * n_rows),
    )
    legend_handles = []
    # Handle the case where axes might be a single element or already a list
    if n_rows == 1 and n_cols == 1:
        axes = [axes]
    elif hasattr(axes, "flatten"):  # Check if axes has flatten method
        axes = axes.flatten()  # Flatten the axes array for easy iteration

    for i, (ax, metric_name) in enumerate(zip(axes, metric_names)):
        metrics_dict = all_metrics_dict[metric_name]
        nan_models = has_nans.get(metric_name, {})
        for j, (model_name, metrics) in enumerate(metrics_dict.items()):
            has_nan = nan_models.get(model_name, False)
            if model_name in model_names_to_exclude:
                continue
            mean_vals = np.array(metrics["means"])
            median_vals = np.array(metrics["medians"])
            all_vals = metrics[
                "all_vals"
            ]  # Keep as list of arrays to avoid inhomogeneous shape error
            if ignore_nans:
                # Filter NaN values from each array in all_vals
                all_vals = [
                    np.array([v for v in val if not np.isnan(v)]) for val in all_vals
                ]
            if metric_name == "spearman" and use_inv_spearman:
                mean_vals = 1 - mean_vals
                median_vals = 1 - median_vals
                all_vals = [1 - val for val in all_vals]

            color = colors[j] if isinstance(colors, list) else colors[model_name]
            if stat_to_plot == "mean":
                ax.plot(
                    metrics["prediction_lengths"],
                    mean_vals,
                    marker=markers[j],
                    label=model_name,
                    markersize=6,
                    color=color,
                    markerfacecolor="none" if has_nan else color,
                    linestyle="-." if has_nan else "-",
                )
                if metric_name in metrics_to_show_envelope:
                    se_envelope = np.array(metrics["stes"])
                    ax.fill_between(
                        metrics["prediction_lengths"],
                        mean_vals - se_envelope,
                        mean_vals + se_envelope,
                        alpha=0.1,
                        color=color,
                    )
            elif stat_to_plot == "median":
                ax.plot(
                    metrics["prediction_lengths"],
                    median_vals,
                    marker=markers[j],
                    label=model_name,
                    markersize=6,
                    color=color,
                    markerfacecolor="none" if has_nan else color,
                    linestyle="-." if has_nan else "-",
                )
                if metric_name in metrics_to_show_envelope:
                    percentile_range_lower = [
                        np.percentile(all_vals[pred_len_idx], percentile_range[0])
                        for pred_len_idx in range(len(all_vals))
                    ]
                    percentile_range_upper = [
                        np.percentile(all_vals[pred_len_idx], percentile_range[1])
                        for pred_len_idx in range(len(all_vals))
                    ]

                    ax.fill_between(
                        metrics["prediction_lengths"],
                        percentile_range_lower,
                        percentile_range_upper,
                        alpha=0.1,
                        color=color,
                    )

        if i == 0:
            legend_handles = [
                plt.Line2D(
                    [0],
                    [0],
                    color=colors[j] if isinstance(colors, list) else colors[model_name],
                    marker=markers[j],
                    markersize=6,
                    label=model_name,
                    linestyle="-." if nan_models.get(model_name, False) else "-",
                    markerfacecolor="none"
                    if nan_models.get(model_name, False)
                    else colors[j]
                    if isinstance(colors, list)
                    else colors[model_name],
                )
                for j, model_name in enumerate(metrics_dict.keys())
            ]
            if show_legend:
                legend_handles = ax.legend(handles=legend_handles, **legend_kwargs)
        ax.set_xlabel("Prediction Length", fontweight="bold", fontsize=12)
        ax.set_xticks(metrics["prediction_lengths"])
        name = metric_name.replace("_", " ")
        if name in ["mse", "mae", "rmse", "mape"]:
            name = name.upper()
        elif name == "smape":
            name = "sMAPE"
        elif name == "spearman" and use_inv_spearman:
            name = "1 - Spearman"
        else:
            name = name.capitalize()
        ax.set_title(name, fontweight="bold", fontsize=16)

    # Hide any unused subplots
    for ax in axes[num_metrics:]:
        ax.set_visible(False)
    if ylim is not None:
        for ax in axes:
            ax.set_ylim(ylim)

    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
    else:
        plt.show()

    return legend_handles
