import json
import os
from multiprocessing import Pool
from typing import Any

from dysts.analysis import max_lyapunov_exponent
from tqdm import tqdm

from panda.utils.dyst_utils import (
    init_base_system_from_params,
    init_skew_system_from_params,
)


def extract_system_data_from_directory(directory: str) -> dict[str, set[int]]:
    """Extract system names and their sample indices from the given directory."""
    system_data = {}

    # Walk through all subdirectories
    for root, dirs, files in os.walk(directory):
        # Extract system name from the directory structure
        # e.g., /path/to/test_zeroshot/Bouali/ -> Bouali
        system_name = os.path.basename(root)

        if system_name in ["test_zeroshot", "test_zeroshot_z5_z10"]:
            continue  # Skip the root directories themselves

        sample_indices = set()
        for file in files:
            # Extract sample index from filename (e.g., "0_T-4096.arrow" -> 0)
            if file.endswith(".arrow"):
                try:
                    sample_idx = int(file.split("_")[0])
                    sample_indices.add(sample_idx)
                except (ValueError, IndexError):
                    continue

        if sample_indices:  # Only add systems that have files
            system_data[system_name] = sample_indices

    return system_data


def filter_successes_by_systems_and_indices(
    successes_data: dict[str, Any], system_data: dict[str, set[int]]
) -> dict[str, Any]:
    """Filter successes data to keep only systems and sample indices that exist in test directories."""
    filtered_data = {}

    for system_name, entries in successes_data.items():
        # Only include systems that exist in the test directories
        if system_name in system_data:
            valid_indices = system_data[system_name]
            filtered_entries = [entry for entry in entries if entry.get("sample_idx") in valid_indices]
            if filtered_entries:
                filtered_data[system_name] = filtered_entries

    return filtered_data


def init_system(params_data: dict[str, Any], system_type: str = "base"):
    init_fn = init_base_system_from_params if system_type == "base" else init_skew_system_from_params
    systems = {}
    for system_name, entries in params_data.items():
        print(f"\nInitializing {system_type} system: {system_name}")
        print(f"Number of parameter sets: {len(entries)}")

        system_list = []
        for i, param_dict in enumerate(entries):
            try:
                if system_type == "skew":
                    driver_name, response_name = system_name.split("_")
                    sys = init_fn(driver_name, response_name, param_dict)
                else:
                    sys = init_fn(system_name, param_dict)
                system_list.append(sys)
            except Exception as e:
                print(f"  Entry {i}: Failed to initialize - {e}")

        if system_list:
            systems[system_name] = system_list

    return systems


def calculate_lyapunov_exponents_worker(args):
    """Worker function for multiprocessing Lyapunov exponent calculation."""
    system_name, dynsys_list = args

    results = []
    for dynsys in dynsys_list:
        try:
            max_lyap = max_lyapunov_exponent(
                eq=dynsys,
                max_walltime=180.0,  # 5 minutes max per trajectory
                rtol=1e-5,
                atol=1e-7,
                n_samples=1,
                traj_length=4096,
            )
            results.append(max_lyap)
        except Exception as e:
            print(f"Error calculating Lyapunov exponent for {system_name}: {e}")
            results.append(None)

    return system_name, results


def calculate_all_lyapunov_exponents(all_systems: dict[str, Any]) -> dict[str, list]:
    """Calculate max Lyapunov exponents for all systems using multiprocessing."""
    print("Calculating max Lyapunov exponents...")

    # Prepare arguments for multiprocessing
    args_list = [(system_name, dynsys_list) for system_name, dynsys_list in all_systems.items()]

    # Use multiprocessing with progress bar
    with Pool() as pool:
        results = list(
            tqdm(
                pool.imap(calculate_lyapunov_exponents_worker, args_list),
                total=len(args_list),
                desc="Processing systems",
            )
        )

    # Convert results to dictionary
    lyapunov_results = {system_name: lyap_values for system_name, lyap_values in results}

    return lyapunov_results


def main():
    work_dir = os.environ.get("WORK", "/stor/work/AMDG_Gilpin_Summer2024")
    base_params_file = f"{work_dir}/data/improved/final_base40/parameters/test/filtered_params_dict.json"
    skew_params_file = f"{work_dir}/data/improved/final_skew40/parameters/test_zeroshot/filtered_params_dict.json"

    with open(base_params_file) as f:
        base_params = json.load(f)

    with open(skew_params_file) as f:
        skew_params = json.load(f)

    print(f"Loaded {len(base_params)} systems from base params")
    print(f"Loaded {len(skew_params)} systems from skew params")

    base_systems = init_system(base_params, "base")
    # skew_systems = init_system(skew_params, "skew")
    # all_systems = {**base_systems, **skew_systems}

    # print(f"Total systems initialized: {len(all_systems)}")

    # Calculate Lyapunov exponents
    # lyapunov_results = calculate_all_lyapunov_exponents(all_systems)
    lyapunov_results = calculate_all_lyapunov_exponents(base_systems)

    # Print summary
    print("\nLyapunov exponent calculation summary:")
    breakpoint()
    for system_name, lyap_values in lyapunov_results.items():
        valid_results = [lyap for lyap in lyap_values if lyap is not None]
        print(f"{system_name}: {len(valid_results)}/{len(lyap_values)} successful calculations")
        if valid_results:
            print(f"  Mean max Lyapunov: {sum(valid_results) / len(valid_results):.4f}")


#     # Define paths
#     work_dir = os.environ.get("WORK", "/stor/work/AMDG_Gilpin_Summer2024")
#     base_dir = f"{work_dir}/data/improved/final_base40"

#     test_zeroshot_dir = f"{base_dir}/test_zeroshot"
#     test_zeroshot_z5_z10_dir = f"{base_dir}/test_zeroshot_z5_z10"
#     successes_file = f"{base_dir}/parameters/test/successes.json"

#     # Extract system data from both directories
#     print("Extracting system data from test_zeroshot...")
#     system_data_1 = extract_system_data_from_directory(test_zeroshot_dir)
#     print(f"Found {len(system_data_1)} systems in test_zeroshot")

#     print("Extracting system data from test_zeroshot_z5_z10...")
#     system_data_2 = extract_system_data_from_directory(test_zeroshot_z5_z10_dir)
#     print(f"Found {len(system_data_2)} systems in test_zeroshot_z5_z10")

#     # Combine system data (merge sample indices for systems that appear in both)
#     combined_system_data = {}
#     all_systems = set(system_data_1.keys()) | set(system_data_2.keys())

#     for system in all_systems:
#         indices_1 = system_data_1.get(system, set())
#         indices_2 = system_data_2.get(system, set())
#         combined_system_data[system] = indices_1 | indices_2

#     print(f"Total unique systems: {len(combined_system_data)}")
#     total_indices = sum(len(indices) for indices in combined_system_data.values())
#     print(f"Total unique sample indices: {total_indices}")

#     # Load successes data
#     print("Loading successes.json...")
#     with open(successes_file, "r") as f:
#         successes_data = json.load(f)

#     print(f"Original successes data has {len(successes_data)} systems")
#     total_entries = sum(len(entries) for entries in successes_data.values())
#     print(f"Original successes data has {total_entries} total entries")

#     # Filter successes data
#     print("Filtering successes data...")
#     filtered_successes = filter_successes_by_systems_and_indices(
#         successes_data, combined_system_data
#     )

#     print(f"Filtered successes data has {len(filtered_successes)} systems")
#     filtered_entries = sum(len(entries) for entries in filtered_successes.values())
#     print(f"Filtered successes data has {filtered_entries} total entries")

#     # Save filtered data
#     output_file = f"{base_dir}/parameters/test/filtered_params_dict.json"
#     print(f"Saving filtered data to {output_file}...")
#     with open(output_file, "w") as f:
#         json.dump(filtered_successes, f, indent=4)

#     print("Done!")


if __name__ == "__main__":
    main()
