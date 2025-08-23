"""
This script computes inheritance metrics for a given model and dataset.

It computes the following metrics:
- KLD between base and skew systems
- KLD between skew systems

See our notebook in notebooks/inheritance.ipynb for more details on use case
"""

import json
import multiprocessing
import os

import numpy as np
from dysts.metrics import estimate_kl_divergence  # type: ignore
from tqdm import tqdm

from panda.utils import (
    load_trajectory_from_arrow,
)

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")


def sample_kld_pairs(pair_type, filepaths_by_dim, num_pairs, rng):
    """
    Randomly sample unique trajectory file pairs for KLD computation.

    Args:
        pair_type (str): "intra" (within system) or "inter" (between systems, same dim).
        filepaths_by_dim (dict): {dim: {system: [filepaths]}}
        num_pairs (int): Number of pairs to sample.
        rng (np.random.Generator): Random number generator.

    Returns:
        list of (filepath_a, filepath_b) tuples.
    """
    # Count total possible pairs without materializing all
    pair_counts = []
    for dim, sysdict in filepaths_by_dim.items():
        systems = list(sysdict)
        if pair_type == "intra":
            for fps in sysdict.values():
                n = len(fps)
                if n >= 2:
                    pair_counts.append(n * (n - 1) // 2)
        elif pair_type == "inter" and len(systems) > 1:
            for i, sys_a in enumerate(systems):
                n_a = len(sysdict[sys_a])
                for sys_b in systems[i + 1 :]:
                    n_b = len(sysdict[sys_b])
                    pair_counts.append(n_a * n_b)
    total_pairs = sum(pair_counts)
    if total_pairs < num_pairs:
        raise ValueError(
            f"Not enough unique {pair_type}-system same-dimension pairs ({total_pairs}) to sample {num_pairs} pairs without repeats."
        )
    chosen_idxs = set(rng.choice(total_pairs, num_pairs, replace=False))
    result, idx_counter = [], 0

    if pair_type == "intra":
        for sysdict in filepaths_by_dim.values():
            for fps in sysdict.values():
                n = len(fps)
                if n < 2:
                    continue
                for i in range(n):
                    for j in range(i + 1, n):
                        if idx_counter in chosen_idxs:
                            result.append((fps[i], fps[j]))
                            if len(result) == num_pairs:
                                return result
                        idx_counter += 1
    elif pair_type == "inter":
        for sysdict in filepaths_by_dim.values():
            systems = list(sysdict)
            for i, sys_a in enumerate(systems):
                fps_a = sysdict[sys_a]
                for sys_b in systems[i + 1 :]:
                    fps_b = sysdict[sys_b]
                    for a in fps_a:
                        for b in fps_b:
                            if idx_counter in chosen_idxs:
                                result.append((a, b))
                                if len(result) == num_pairs:
                                    return result
                            idx_counter += 1
    return result


def compute_klds(pairs):
    klds = []
    # for file_a, file_b in tqdm(pairs, desc="Computing KLDs"):
    for file_a, file_b in pairs:
        coords_a, _ = load_trajectory_from_arrow(file_a)
        coords_b, _ = load_trajectory_from_arrow(file_b)
        # print(f"Shape of coords_a: {coords_a.shape}, coords_b: {coords_b.shape}")
        if coords_a.shape[0] != coords_b.shape[0]:
            print(
                f"Skipping pair due to mismatched dimensions: {coords_a.shape[0]} vs {coords_b.shape[0]}"
            )
            continue
        kld = estimate_kl_divergence(coords_a.T, coords_b.T)
        # print(f"KLD: {kld}")
        klds.append(kld)
    return klds


def compute_klds_for_pair(pair):
    # compute_klds expects a list of pairs, so wrap in a list
    return compute_klds([pair]) or []


def gather_filepaths_by_dim_and_system(root_dir, system_names, desc=None):
    """Return {dim: {system: [filepaths]}} for given systems in root_dir."""
    filepaths = {}
    iterator = tqdm(system_names, desc=desc) if desc else system_names
    for system in iterator:
        subdir = os.path.join(root_dir, system)
        for file in sorted(os.listdir(subdir)):
            coords, _ = load_trajectory_from_arrow(os.path.join(subdir, file))
            dim = coords.shape[0]
            filepaths.setdefault(dim, {}).setdefault(system, []).append(
                os.path.join(subdir, file)
            )
    return filepaths


def parse_driver_response(skew_name):
    return tuple(skew_name.split("_", 1)) if "_" in skew_name else (skew_name, None)


# def sample_skew_vs_base_pairs(skew_filepaths, base_filepaths, which, num_pairs, rng):
#     """
#     which: "driver", "response", or "base"
#     For "driver" or "response", pairs skew system with its driver/response base system.
#     For "base", pairs skew system with a base system that is neither its driver nor response.
#     """
#     pairs = []
#     for dim, skew_dim_dict in skew_filepaths.items():
#         base_dim_dict = base_filepaths.get(dim)
#         if not base_dim_dict:
#             continue
#         for skew_name, skew_files in skew_dim_dict.items():
#             driver, response = parse_driver_response(skew_name)
#             if which in ("driver", "response"):
#                 base_name = driver if which == "driver" else response
#                 if not base_name or base_name not in base_dim_dict:
#                     continue
#                 base_files = base_dim_dict[base_name]
#                 n = min(num_pairs, len(skew_files), len(base_files))
#                 if n == 0:
#                     continue
#                 pairs.extend(
#                     zip(
#                         list(
#                             np.array(skew_files)[
#                                 rng.choice(len(skew_files), n, replace=False)
#                             ]
#                         )
#                         if len(skew_files) > n
#                         else skew_files,
#                         list(
#                             np.array(base_files)[
#                                 rng.choice(len(base_files), n, replace=False)
#                             ]
#                         )
#                         if len(base_files) > n
#                         else base_files,
#                     )
#                 )
#             elif which == "base":
#                 # Exclude driver and response from base candidates
#                 exclude = {driver, response}
#                 base_candidates = [
#                     name
#                     for name in base_dim_dict
#                     if name not in exclude and name is not None
#                 ]
#                 if not base_candidates:
#                     continue
#                 base_name = rng.choice(base_candidates)
#                 base_files = base_dim_dict[base_name]
#                 n = min(num_pairs, len(skew_files), len(base_files))
#                 if n == 0:
#                     continue
#                 pairs.extend(
#                     zip(
#                         list(
#                             np.array(skew_files)[
#                                 rng.choice(len(skew_files), n, replace=False)
#                             ]
#                         )
#                         if len(skew_files) > n
#                         else skew_files,
#                         list(
#                             np.array(base_files)[
#                                 rng.choice(len(base_files), n, replace=False)
#                             ]
#                         )
#                         if len(base_files) > n
#                         else base_files,
#                     )
#                 )
#     if len(pairs) > num_pairs:
#         idxs = rng.choice(len(pairs), num_pairs, replace=False)
#         return [pairs[i] for i in idxs]
#     else:
#         return pairs


def sample_skew_vs_base_pairs(skew_filepaths, base_filepaths, which, num_pairs, rng):
    """
    which: "driver", "response", "base", "skew_intra", or "skew_inter"
    For "driver" or "response", pairs skew system with its driver/response base system.
    For "base", pairs skew system with a base system that is neither its driver nor response.
    For "skew_intra", pairs skew system with another skew system (intra-system).
    For "skew_inter", pairs skew system with another skew system (inter-system).
    """
    pairs = []
    if which == "skew_intra":
        # Only intra-system pairs (within the same skew system)
        intra_pairs = []
        for dim, skew_dim_dict in skew_filepaths.items():
            for skew_name, skew_files in skew_dim_dict.items():
                n = len(skew_files)
                if n >= 2:
                    all_pairs = [
                        (skew_files[i], skew_files[j])
                        for i in range(n)
                        for j in range(i + 1, n)
                    ]
                    intra_pairs.extend(all_pairs)
        if len(intra_pairs) > num_pairs:
            idxs = rng.choice(len(intra_pairs), num_pairs, replace=False)
            intra_pairs = [intra_pairs[i] for i in idxs]
        return intra_pairs
    elif which == "skew_inter":
        # Only inter-system pairs (between different skew systems, same dimension)
        inter_pairs = []
        for dim, skew_dim_dict in skew_filepaths.items():
            skew_systems = list(skew_dim_dict)
            if len(skew_systems) > 1:
                for i, sys_a in enumerate(skew_systems):
                    files_a = skew_dim_dict[sys_a]
                    for sys_b in skew_systems[i + 1 :]:
                        files_b = skew_dim_dict[sys_b]
                        inter_pairs.extend([(a, b) for a in files_a for b in files_b])
        if len(inter_pairs) > num_pairs:
            idxs = rng.choice(len(inter_pairs), num_pairs, replace=False)
            inter_pairs = [inter_pairs[i] for i in idxs]
        return inter_pairs
    else:
        for dim, skew_dim_dict in skew_filepaths.items():
            base_dim_dict = base_filepaths.get(dim)
            if not base_dim_dict:
                continue
            for skew_name, skew_files in skew_dim_dict.items():
                driver, response = parse_driver_response(skew_name)
                if which in ("driver", "response"):
                    base_name = driver if which == "driver" else response
                    if not base_name or base_name not in base_dim_dict:
                        continue
                    base_files = base_dim_dict[base_name]
                    n = min(num_pairs, len(skew_files), len(base_files))
                    if n == 0:
                        continue
                    if len(skew_files) > n:
                        skew_idxs = rng.choice(len(skew_files), n, replace=False)
                        skew_sample = [skew_files[i] for i in skew_idxs]
                    else:
                        skew_sample = skew_files
                    if len(base_files) > n:
                        base_idxs = rng.choice(len(base_files), n, replace=False)
                        base_sample = [base_files[i] for i in base_idxs]
                    else:
                        base_sample = base_files
                    pairs.extend(zip(skew_sample, base_sample))
                elif which == "base":
                    # Exclude driver and response from base candidates
                    exclude = {driver, response}
                    base_candidates = [
                        name
                        for name in base_dim_dict
                        if name not in exclude and name is not None
                    ]
                    if not base_candidates:
                        continue
                    base_name = rng.choice(base_candidates)
                    base_files = base_dim_dict[base_name]
                    n = min(num_pairs, len(skew_files), len(base_files))
                    if n == 0:
                        continue
                    if len(skew_files) > n:
                        skew_idxs = rng.choice(len(skew_files), n, replace=False)
                        skew_sample = [skew_files[i] for i in skew_idxs]
                    else:
                        skew_sample = skew_files
                    if len(base_files) > n:
                        base_idxs = rng.choice(len(base_files), n, replace=False)
                        base_sample = [base_files[i] for i in base_idxs]
                    else:
                        base_sample = base_files
                    pairs.extend(zip(skew_sample, base_sample))
        if len(pairs) > num_pairs:
            idxs = rng.choice(len(pairs), num_pairs, replace=False)
            return [pairs[i] for i in idxs]
        else:
            return pairs


def base(num_base_systems: int, num_pairs: int, save_fname_suffix: str | None = None):
    # Sample base systems and gather filepaths by dimension and system
    base_dir = os.path.join(DATA_DIR, base_split_name)
    base_system_names = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    sampled_base_systems = list(
        np.array(base_system_names)[
            rng.choice(
                len(base_system_names),
                size=min(num_base_systems, len(base_system_names)),
                replace=False,
            )
        ]
    )

    base_filepaths = {}
    for system in sampled_base_systems:
        subdir = os.path.join(base_dir, system)
        for file in sorted(os.listdir(subdir)):
            filepath = os.path.join(subdir, file)
            coords, _ = load_trajectory_from_arrow(filepath)
            dim = coords.shape[0]
            base_filepaths.setdefault(dim, {}).setdefault(system, []).append(filepath)

    # Compute KLDs for intra- and inter-system pairs, store results in a dict
    base_kld_results = {}

    for pair_type in ["intra", "inter"]:
        pairs = sample_kld_pairs(pair_type, base_filepaths, num_pairs, rng)

        with multiprocessing.Pool(processes=100) as pool:
            klds = list(
                tqdm(
                    pool.imap(compute_klds_for_pair, pairs),
                    total=len(pairs),
                    desc=f"KLD {pair_type} pairs",
                    leave=False,
                )
            )

        if klds:
            base_kld_results[pair_type] = {
                "pairs": pairs,
                "mean": np.mean(klds),
                "std": np.std(klds),
                "values": klds,
            }
        else:
            base_kld_results[pair_type] = {
                "pairs": pairs,
                "mean": None,
                "std": None,
                "values": [],
            }

    # Optionally print concise summary
    for pair_type, res in base_kld_results.items():
        print(
            f"{pair_type.capitalize()}-system base pairs: mean KLD={res['mean']}, std={res['std']}, n={len(res['values'])}"
        )

    # Print concise summary for base system KLDs
    for pair_type, res in base_kld_results.items():
        print(
            f"{pair_type.capitalize()}-system base pairs: mean KLD={res['mean']}, std={res['std']}, n={len(res['values'])}"
        )

    if save_fname_suffix is None:
        save_fname_suffix = ""
    output_json_path = os.path.join(
        "outputs/inheritance",
        f"{base_split_name}_kld_results{save_fname_suffix}.json",
    )
    os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

    # Convert any numpy types to native Python types for JSON serialization
    def convert_np(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating)):
            return float(obj)
        if isinstance(obj, (np.integer)):
            return int(obj)
        return obj

    with open(output_json_path, "w") as f:
        json.dump(base_kld_results, f, default=convert_np, indent=4)
    print(f"Dumped base KLD results to {output_json_path}")


def skew(num_skew_systems: int, num_pairs: int, save_fname_suffix: str | None = None):
    skew_dir = os.path.join(DATA_DIR, skew_split_name)

    skew_system_names = [
        d for d in os.listdir(skew_dir) if os.path.isdir(os.path.join(skew_dir, d))
    ]
    sampled_skew_systems = rng.choice(
        skew_system_names,
        size=min(num_skew_systems, len(skew_system_names)),
        replace=False,
    ).tolist()

    # Gather filepaths for base and skew systems with progress bars
    base_dir = os.path.join(DATA_DIR, base_split_name)
    base_system_names = [
        d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))
    ]
    base_filepaths = gather_filepaths_by_dim_and_system(
        base_dir, base_system_names, desc="Base systems"
    )
    skew_filepaths = gather_filepaths_by_dim_and_system(
        skew_dir, sampled_skew_systems, desc="Skew systems"
    )
    skew_kld_results = {}

    for which in ["skew_intra", "skew_inter"]:
        # for which in ["driver", "response", "base", "skew"]:
        if which == "skew_intra":
            print("Computing KLDs for skew-skew intra-system pairs...")
        elif which == "skew_inter":
            print("Computing KLDs for skew-skew inter-system pairs...")
        else:
            print(f"Computing KLDs for skew-{which} vs. base system pairs...")
        pairs = sample_skew_vs_base_pairs(
            skew_filepaths, base_filepaths, which, num_pairs, rng
        )
        if pairs:
            with multiprocessing.Pool(processes=100) as pool:
                # Map each pair to its KLD(s)
                results = list(
                    tqdm(
                        pool.imap(compute_klds_for_pair, pairs),
                        total=len(pairs),
                        desc=f"KLD skew-{which} pairs",
                        leave=False,
                    )
                )
            # Flatten the list of lists
            klds = [kld for sublist in results for kld in sublist]
        else:
            klds = []
        skew_kld_results[which] = {
            "pairs": pairs,
            "mean": np.mean(klds) if klds else None,
            "std": np.std(klds) if klds else None,
            "values": klds,
        }

    # Print concise summary
    for which, res in skew_kld_results.items():
        print(
            f"Skew-{which} vs. base system pairs: mean KLD={res['mean']}, std={res['std']}, n={len(res['values'])}"
        )

        if save_fname_suffix is None:
            save_fname_suffix = ""
        output_json_path = os.path.join(
            "outputs/inheritance",
            f"{skew_split_name}_kld_results{save_fname_suffix}.json",
        )
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        # Convert any numpy types to native Python types for JSON serialization
        def convert_np(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.floating)):
                return float(obj)
            if isinstance(obj, (np.integer)):
                return int(obj)
            return obj

        with open(output_json_path, "w") as f:
            json.dump(skew_kld_results, f, default=convert_np, indent=4)
        print(f"Dumped skew KLD results to {output_json_path}")


if __name__ == "__main__":
    skew_split_name = "improved/final_skew40/train"
    base_split_name = "improved/final_base40/train"

    rseed = 987
    rng = np.random.default_rng(rseed)
    # base(num_base_systems=111, num_pairs=4000, save_fname_suffix=f"_rseed{rseed}")
    skew(num_skew_systems=1109, num_pairs=10000, save_fname_suffix=f"_rseed{rseed}")
