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

from dystformer.utils.data_utils import stack_and_extract_metadata
from dystformer.utils.plot_utils import plot_trajs_multivariate, plot_univariate_trajs


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
        # Get the magnitude of the FFT
        fft_vals = np.fft.rfft(ts[dim])
        magnitudes = np.abs(fft_vals)
        # Get the corresponding frequencies
        freqs = np.fft.rfftfreq(ts.shape[1])
        freq_spectra.append((freqs, magnitudes))

    return freq_spectra


def plot_fft_spectrum(ts: np.ndarray, max_dims: int = 3) -> plt.Figure:
    """
    Plot FFT spectrum for a time series.

    Args:
        ts: Time series array of shape (D, T)
        max_dims: Maximum number of dimensions to plot

    Returns:
        Matplotlib figure
    """
    dims_to_plot = min(ts.shape[0], max_dims)
    spectra = compute_fft_spectrum(ts)

    fig, axes = plt.subplots(dims_to_plot, 1, figsize=(10, 3 * dims_to_plot))
    if dims_to_plot == 1:
        axes = [axes]

    for i in range(dims_to_plot):
        freqs, mags = spectra[i]
        axes[i].plot(freqs, mags)
        axes[i].set_title(f"Dimension {i + 1} Frequency Spectrum")
        axes[i].set_xlabel("Frequency")
        axes[i].set_ylabel("Magnitude")
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
    time_series, _ = stack_and_extract_metadata(dataset)
    return time_series


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
    """Create all required plots for the time series."""
    # Create a temporary directory for plot_trajs_multivariate to save to
    temp_dir = Path("temp_plots")
    temp_dir.mkdir(exist_ok=True)

    # 3D Plot
    plot_trajs_multivariate(
        np.expand_dims(time_series, 0),  # Add batch dimension
        save_dir=str(temp_dir),
        plot_name="current_series",
        plot_2d_slice=True,
        plot_projections=True,
        figsize=(10, 8),
    )

    # Load the generated plots
    fig_3d = plt.figure(figsize=(10, 8))
    img_3d = plt.imread(temp_dir / "current_series_3D.png")
    plt.imshow(img_3d)
    plt.axis("off")

    # 1D univariate plots
    univariate_fig = plt.figure(figsize=(10, 8))
    time_series_dict = {"current_series": np.expand_dims(time_series, 0)}
    plot_univariate_trajs(
        time_series_dict, save_path=str(temp_dir), figsize=(10, 8), standardize=False
    )

    univariate_img = plt.imread(temp_dir / "current_series_coords.png")
    plt.imshow(univariate_img)
    plt.axis("off")

    # FFT spectrum
    fft_fig = plot_fft_spectrum(time_series)

    return fig_3d, univariate_fig, fft_fig


def get_system_name(file_path: Path) -> str:
    """Extract the system name from the file path."""
    # The system name is the parent directory name
    return file_path.parent.name


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
                    all_files = get_arrow_files_recursive(data_dir)
                    if not all_files:
                        st.error(
                            "No .arrow files found in the specified directory structure"
                        )
                    else:
                        st.session_state.files = all_files
                        st.session_state.current_idx = 0
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

    # Main content
    if "files" in st.session_state and st.session_state.files:
        col1, col2 = st.columns([3, 1])

        with col1:
            current_file = st.session_state.files[st.session_state.current_idx]
            system_name = get_system_name(current_file)

            st.subheader(f"System: {system_name}")
            st.subheader(f"File: {current_file.name}")

            # Load and display time series
            with st.spinner("Loading time series..."):
                time_series = load_time_series(current_file)

                # Create tabs for different visualizations
                tab1, tab2, tab3 = st.tabs(
                    ["3D Plot", "Univariate Plots", "FFT Spectrum"]
                )

                with tab1:
                    fig_3d, _, _ = create_time_series_plots(time_series)
                    st.pyplot(fig_3d)

                with tab2:
                    _, univariate_fig, _ = create_time_series_plots(time_series)
                    st.pyplot(univariate_fig)

                with tab3:
                    _, _, fft_fig = create_time_series_plots(time_series)
                    st.pyplot(fft_fig)

        with col2:
            st.markdown("### Evaluation")
            st.write("Is this time series good?")

            col_yes, col_no = st.columns(2)

            with col_yes:
                if st.button("👍 Yes", use_container_width=True):
                    save_evaluation(str(current_file), True, output_file)
                    if st.session_state.current_idx < len(st.session_state.files) - 1:
                        st.session_state.current_idx += 1
                        st.rerun()
                    else:
                        st.success("All files have been evaluated!")

            with col_no:
                if st.button("👎 No", use_container_width=True):
                    save_evaluation(str(current_file), False, output_file)
                    if st.session_state.current_idx < len(st.session_state.files) - 1:
                        st.session_state.current_idx += 1
                        st.rerun()
                    else:
                        st.success("All files have been evaluated!")

            # Navigation buttons
            st.markdown("### Navigation")
            nav_col1, nav_col2 = st.columns(2)

            with nav_col1:
                if st.button("⬅️ Previous", use_container_width=True):
                    if st.session_state.current_idx > 0:
                        st.session_state.current_idx -= 1
                        st.rerun()

            with nav_col2:
                if st.button("➡️ Next", use_container_width=True):
                    if st.session_state.current_idx < len(st.session_state.files) - 1:
                        st.session_state.current_idx += 1
                        st.rerun()

            # Random jump option
            if st.button("🔀 Random Sample", use_container_width=True):
                import random

                st.session_state.current_idx = random.randint(
                    0, len(st.session_state.files) - 1
                )
                st.rerun()

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
    else:
        st.info("Please load .arrow files from the sidebar to start evaluation")


if __name__ == "__main__":
    main()
