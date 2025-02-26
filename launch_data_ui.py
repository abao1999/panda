import os
import sys

print("Python executable:", sys.executable)
print("Python path:", sys.path)

import csv
from pathlib import Path
from typing import List, Tuple

import matplotlib.pyplot as plt
import numpy as np
import streamlit as st
from gluonts.dataset.common import FileDataset


def compute_fft_spectrum(ts: np.ndarray) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    Compute FFT spectrum for a time series.

    Args:
        ts: Time series array of shape (D, T)

    Returns:
        List of tuples (frequencies, magnitudes) for each dimension
    """
    # Apply FFT to each dimension
    freq_spectra = []
    for dim in range(ts.shape[0]):
        # Get the FFT values
        fft_vals = np.fft.rfft(ts[dim])
        # Compute power spectrum (squared magnitude)
        magnitudes = np.abs(fft_vals) ** 2
        # Get the corresponding frequencies
        freqs = np.fft.rfftfreq(ts.shape[1])
        freq_spectra.append((freqs, magnitudes))

    return freq_spectra


def plot_fft_spectrum(ts: np.ndarray, max_dims: int = None) -> plt.Figure:
    """
    Plot FFT spectrum for a time series.

    Args:
        ts: Time series array of shape (D, T)
        max_dims: Maximum number of dimensions to plot, None means all dimensions

    Returns:
        Matplotlib figure
    """
    # If max_dims is None, plot all dimensions
    dims_to_plot = ts.shape[0] if max_dims is None else min(ts.shape[0], max_dims)
    spectra = compute_fft_spectrum(ts)

    fig, axes = plt.subplots(dims_to_plot, 1, figsize=(10, 2.5 * dims_to_plot))
    if dims_to_plot == 1:
        axes = [axes]

    for i in range(dims_to_plot):
        freqs, mags = spectra[i]
        axes[i].semilogy(freqs, mags)
        axes[i].set_title(f"Dimension {i + 1} Frequency Spectrum")
        axes[i].set_xlabel("Frequency")
        axes[i].set_ylabel("Magnitude (log scale)")
        axes[i].set_xlim(0, 0.5)  # Nyquist frequency is 0.5
        axes[i].grid(True)

    plt.tight_layout()
    return fig


def get_arrow_files_recursive(base_dir: str) -> List[Path]:
    """
    Recursively get all arrow files from the base directory.
    Handles the hierarchical structure of system directories containing samples.
    """
    base_path = Path(base_dir)
    if not base_path.exists():
        return []

    # Find all .arrow files recursively
    all_files = sorted(list(base_path.glob("**/*.arrow")))
    return all_files


def load_time_series(file_path: Path) -> np.ndarray:
    """Load time series from arrow file."""
    dataset = FileDataset(path=file_path, one_dim_target=False, freq="h")
    return next(iter(dataset))["target"]


def save_evaluation(
    file_path: str, is_good: bool, output_file: str = "evaluations.csv"
):
    """Save evaluation result to CSV file."""
    file_exists = os.path.isfile(output_file)

    with open(output_file, "a", newline="") as csvfile:
        fieldnames = ["file_path", "evaluation"]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)

        if not file_exists:
            writer.writeheader()

        writer.writerow(
            {"file_path": str(file_path), "evaluation": "good" if is_good else "bad"}
        )


def create_time_series_plots(
    time_series: np.ndarray,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Create all required plots for the time series without saving to disk."""
    # For 3D plot, create a figure directly instead of using plot_trajs_multivariate
    fig_3d = plt.figure(figsize=(10, 8))
    ax = fig_3d.add_subplot(111, projection="3d")

    # Plot the first three dimensions
    dims_3d = [0, 1, 2]
    xyz = time_series[dims_3d, :]
    ax.plot(*xyz, alpha=0.5, linewidth=1)

    # Initial and final points
    ic_pt = xyz[:, 0]
    ax.scatter(*ic_pt, marker="*", s=100, alpha=0.5)

    end_pt = xyz[:, -1]
    ax.scatter(*end_pt, marker="x", s=100, alpha=0.5)

    ax.set_xlabel(f"dim_{dims_3d[0]}")
    ax.set_ylabel(f"dim_{dims_3d[1]}")
    ax.set_zlabel(f"dim_{dims_3d[2]}")

    # For univariate plots - show all dimensions
    univariate_fig = plt.figure(figsize=(10, min(2.5 * time_series.shape[0], 15)))
    for i in range(time_series.shape[0]):  # Display all dimensions
        plt.subplot(time_series.shape[0], 1, i + 1)
        plt.plot(time_series[i, :])
        plt.ylabel(f"Dim {i + 1}")
    plt.tight_layout()

    # FFT spectrum - pass None to plot all dimensions
    fft_fig = plot_fft_spectrum(time_series, max_dims=None)

    return fig_3d, univariate_fig, fft_fig


def get_system_name(file_path: Path) -> str:
    """Extract the system name from the file path."""
    # The system name is the parent directory name
    return file_path.parent.name


@st.cache_data(ttl="1h")
def cached_get_arrow_files_recursive(base_dir: str) -> List[Path]:
    """Cached version of get_arrow_files_recursive to avoid repeated filesystem operations"""
    return get_arrow_files_recursive(base_dir)


@st.cache_data(ttl="1h")
def cached_load_time_series(file_path: str) -> np.ndarray:
    """Cached version of load_time_series to avoid repeated file loading"""
    dataset = FileDataset(path=Path(file_path), one_dim_target=False, freq="h")
    return next(iter(dataset))["target"]


@st.cache_data(ttl="1h", max_entries=10)
def cached_create_plots(
    time_series: np.ndarray,
) -> Tuple[plt.Figure, plt.Figure, plt.Figure]:
    """Cached version of plot creation to avoid regenerating the same plots"""
    return create_time_series_plots(time_series)


def main():
    st.set_page_config(layout="wide", page_title="Time Series Evaluator")

    # Server configuration
    if "server_config" not in st.session_state:
        st.session_state.server_config = True

        os.environ["STREAMLIT_SERVER_PORT"] = "8501"  # Or any port you prefer
        os.environ["STREAMLIT_SERVER_HEADLESS"] = "true"
        os.environ["STREAMLIT_SERVER_ADDRESS"] = "0.0.0.0"  # Listen on all interfaces

    st.title("Time Series Evaluator")

    # Path input
    with st.sidebar:
        st.header("Input Settings")
        data_dir = st.text_input(
            "Enter path to base directory containing system folders with .arrow files",
            help="Directory should contain system subdirectories, each with .arrow files",
        )

        if st.button("Load Files"):
            if not data_dir or not os.path.isdir(data_dir):
                st.error("Please enter a valid directory path")
            else:
                with st.spinner("Loading arrow files..."):
                    all_files = cached_get_arrow_files_recursive(data_dir)
                    if not all_files:
                        st.error(
                            "No .arrow files found in the specified directory structure"
                        )
                    else:
                        st.session_state.files = all_files
                        st.session_state.current_idx = 0
                        st.session_state.preloaded_data = {}  # For prefetching
                        st.success(f"Loaded {len(all_files)} .arrow files")

        output_file = st.text_input("Output CSV file", "evaluations.csv")

        # Display progress
        if "files" in st.session_state:
            st.write(
                f"Progress: {st.session_state.current_idx + 1}/{len(st.session_state.files)}"
            )

            # Add system filter option
            if "system_names" not in st.session_state:
                system_names = sorted(
                    list(set(get_system_name(f) for f in st.session_state.files))
                )
                st.session_state.system_names = system_names
                st.session_state.system_filter = "All"

            system_filter = st.selectbox(
                "Filter by system:", ["All"] + st.session_state.system_names
            )

            if system_filter != st.session_state.system_filter:
                st.session_state.system_filter = system_filter
                # Update filtered files
                if system_filter == "All":
                    # Keep all files
                    pass
                else:
                    # Filter to only show files from the selected system
                    filtered_files = [
                        f
                        for f in st.session_state.files
                        if get_system_name(f) == system_filter
                    ]
                    if filtered_files:
                        st.session_state.files = filtered_files
                        st.session_state.current_idx = 0
                        st.success(
                            f"Filtered to {len(filtered_files)} files from {system_filter}"
                        )
                    else:
                        st.error(f"No files found for system {system_filter}")

    # Main content - use this block design to prevent cascading reruns
    if "files" in st.session_state and st.session_state.files:
        # Use a container to manage reflows better
        main_container = st.container()

        # Define all columns once
        with main_container:
            col1, col2 = st.columns([3, 1])

        # Get current file info - do this outside of any column
        current_file = st.session_state.files[st.session_state.current_idx]
        system_name = get_system_name(current_file)

        # Load time series (with caching) - do this outside of any column
        time_series = cached_load_time_series(str(current_file))

        # Get cached plots - do this outside of any column
        fig_3d, univariate_fig, fft_fig = cached_create_plots(time_series)

        # Only now populate the columns with content
        with col1:
            st.subheader(f"System: {system_name}")
            st.subheader(f"File: {current_file.name}")

            # Create tabs
            tab1, tab2, tab3 = st.tabs(["3D Plot", "Univariate Plots", "FFT Spectrum"])

            with tab1:
                st.pyplot(fig_3d)

            with tab2:
                st.pyplot(univariate_fig)

            with tab3:
                st.pyplot(fft_fig)

        # Now handle the second column
        with col2:
            # Custom vertical spacing using HTML and CSS
            st.markdown(
                """
                <div style="height: 65px;"></div>
                """,
                unsafe_allow_html=True,
            )

            st.markdown("### Evaluation")
            st.markdown(
                """
                <hr style="height:2px;border-width:0;color:gray;background-color:gray">
                """,
                unsafe_allow_html=True,
            )

            # Define the button callbacks before creating the buttons
            def on_yes_click():
                save_evaluation(str(current_file), True, output_file)
                if st.session_state.current_idx < len(st.session_state.files) - 1:
                    st.session_state.current_idx += 1

                    # Prefetch next file here, before rerun
                    if st.session_state.current_idx < len(st.session_state.files) - 1:
                        next_file = st.session_state.files[
                            st.session_state.current_idx + 1
                        ]
                        cached_load_time_series(str(next_file))

                    # Use session state to track button clicks instead of rerunning immediately
                    st.session_state.pending_rerun = True
                else:
                    st.success("All files have been evaluated!")

            def on_no_click():
                save_evaluation(str(current_file), False, output_file)
                if st.session_state.current_idx < len(st.session_state.files) - 1:
                    st.session_state.current_idx += 1

                    # Prefetch next file here, before rerun
                    if st.session_state.current_idx < len(st.session_state.files) - 1:
                        next_file = st.session_state.files[
                            st.session_state.current_idx + 1
                        ]
                        cached_load_time_series(str(next_file))

                    # Use session state to track button clicks
                    st.session_state.pending_rerun = True
                else:
                    st.success("All files have been evaluated!")

            # Set up key-based buttons to avoid interference between reruns
            btn_idx = st.session_state.current_idx

            col_yes, col_no = st.columns(2)
            with col_yes:
                if st.button(
                    "👍 Yes",
                    key=f"yes_{btn_idx}",
                    on_click=on_yes_click,
                    use_container_width=True,
                ):
                    pass  # The on_click handler does everything

            with col_no:
                if st.button(
                    "👎 No",
                    key=f"no_{btn_idx}",
                    on_click=on_no_click,
                    use_container_width=True,
                ):
                    pass  # The on_click handler does everything

            # Navigation buttons
            st.markdown("### Navigation")
            nav_col1, nav_col2 = st.columns(2)

            with nav_col1:
                if st.button(
                    "⬅️ Previous", key=f"prev_{btn_idx}", use_container_width=True
                ):
                    if st.session_state.current_idx > 0:
                        st.session_state.current_idx -= 1
                        st.session_state.pending_rerun = True

            with nav_col2:
                if st.button("➡️ Next", key=f"next_{btn_idx}", use_container_width=True):
                    if st.session_state.current_idx < len(st.session_state.files) - 1:
                        st.session_state.current_idx += 1
                        st.session_state.pending_rerun = True

            # Random jump option
            if st.button(
                "🔀 Random Sample", key=f"random_{btn_idx}", use_container_width=True
            ):
                import random

                st.session_state.current_idx = random.randint(
                    0, len(st.session_state.files) - 1
                )
                st.session_state.pending_rerun = True

            # Display file metadata
            st.markdown("### File Information")
            st.write(f"**System:** {system_name}")
            st.write(f"**Path:** {current_file}")
            st.write(f"**Shape:** {time_series.shape}")

            # Display summary statistics for each dimension
            st.markdown("### Summary Statistics")
            stats_data = {
                "Min": np.min(time_series, axis=1),
                "Max": np.max(time_series, axis=1),
                "Mean": np.mean(time_series, axis=1),
                "Std": np.std(time_series, axis=1),
            }

            # Format data for display
            stats_rows = []
            for i in range(
                min(time_series.shape[0], 5)
            ):  # Display stats for first 5 dimensions
                stats_rows.append(
                    {
                        "Dim": f"Dim {i + 1}",
                        "Min": f"{stats_data['Min'][i]:.4f}",
                        "Max": f"{stats_data['Max'][i]:.4f}",
                        "Mean": f"{stats_data['Mean'][i]:.4f}",
                        "Std": f"{stats_data['Std'][i]:.4f}",
                    }
                )

            st.table(stats_rows)

            # Prefetch next file data in the background if available
            if st.session_state.current_idx < len(st.session_state.files) - 1:
                next_file = st.session_state.files[st.session_state.current_idx + 1]
                # This will cache the next file's data without displaying it
                cached_load_time_series(str(next_file))

    # Handle pending reruns at the end of the script - this prevents cascading reruns
    if "pending_rerun" in st.session_state and st.session_state.pending_rerun:
        st.session_state.pending_rerun = False
        st.rerun()


if __name__ == "__main__":
    main()
