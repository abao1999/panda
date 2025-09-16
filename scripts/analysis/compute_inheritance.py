"""
This script computes inheritance metrics for a given model and dataset.

It computes the following metrics:
- KLD between base and skew systems
- KLD between skew systems

See our notebook in notebooks/inheritance.ipynb for more details on use case.
"""

import json
import multiprocessing
import os
from collections.abc import Sequence
from typing import Any

import numpy as np
from dysts.metrics import estimate_kl_divergence  # type: ignore
from tqdm import tqdm

from panda.utils.data_utils import load_trajectory_from_arrow

WORK_DIR: str = os.getenv("WORK", "")
DATA_DIR: str = os.path.join(WORK_DIR, "data")


def sample_kld_pairs(
    pair_type: str,
    filepaths_by_dim: dict[int, dict[str, list[str]]],
    num_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """
    Randomly sample unique trajectory file pairs for KLD computation.

    Args:
        pair_type: "intra" (within system) or "inter" (between systems, same dimension).
        filepaths_by_dim: Mapping from dimension to {system: [filepaths]}.
        num_pairs: Number of pairs to sample.
        rng: Numpy random number generator.

    Returns:
        List of (filepath_a, filepath_b) tuples representing sampled pairs.
    Raises:
        ValueError: If not enough unique pairs are available to sample without repeats.
    """
    pair_counts: list[int] = []
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
    total_pairs: int = sum(pair_counts)
    if total_pairs < num_pairs:
        raise ValueError(
            f"Not enough unique {pair_type}-system same-dimension pairs ({total_pairs}) to sample {num_pairs} pairs without repeats."
        )
    chosen_idxs: set[int] = set(rng.choice(total_pairs, num_pairs, replace=False))
    result: list[tuple[str, str]] = []
    idx_counter: int = 0

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


def compute_klds(pairs: Sequence[tuple[str, str]]) -> list[float]:
    """
    Compute Kullback-Leibler divergences for a list of trajectory file pairs.

    Args:
        pairs: Sequence of (filepath_a, filepath_b) tuples.

    Returns:
        List of KLD values (float) for each valid pair.
    """
    klds: list[float] = []
    for file_a, file_b in pairs:
        coords_a, _ = load_trajectory_from_arrow(file_a)
        coords_b, _ = load_trajectory_from_arrow(file_b)
        if coords_a.shape[0] != coords_b.shape[0]:
            print(f"Skipping pair due to mismatched dimensions: {coords_a.shape[0]} vs {coords_b.shape[0]}")
            continue
        kld = estimate_kl_divergence(coords_a.T, coords_b.T)
        klds.append(kld)
    return klds


def compute_klds_for_pair(pair: tuple[str, str]) -> list[float]:
    """
    Compute KLD for a single pair of trajectory files.

    Args:
        pair: Tuple of (filepath_a, filepath_b).

    Returns:
        List containing a single KLD value, or empty if invalid.
    """
    return compute_klds([pair]) or []


def gather_filepaths_by_dim_and_system(
    root_dir: str,
    system_names: Sequence[str],
    desc: str | None = None,
) -> dict[int, dict[str, list[str]]]:
    """
    Gather trajectory filepaths organized by dimension and system.

    Args:
        root_dir: Root directory containing system subdirectories.
        system_names: List of system names (subdirectory names).
        desc: Optional description for tqdm progress bar.

    Returns:
        Dictionary mapping dimension -> {system: [filepaths]}.
    """
    filepaths: dict[int, dict[str, list[str]]] = {}
    iterator = tqdm(system_names, desc=desc) if desc else system_names
    for system in iterator:
        subdir = os.path.join(root_dir, system)
        for file in sorted(os.listdir(subdir)):
            coords, _ = load_trajectory_from_arrow(os.path.join(subdir, file))
            dim = coords.shape[0]
            filepaths.setdefault(dim, {}).setdefault(system, []).append(os.path.join(subdir, file))
    return filepaths


def parse_driver_response(
    skew_name: str,
) -> tuple[str, str | None]:
    """
    Parse a skew system name into driver and response components.

    Args:
        skew_name: Name of the skew system, e.g., "driver_response" or "driver".

    Returns:
        Tuple of (driver, response) where response may be None (in the case where the system is not a skew system)
    """
    return tuple(skew_name.split("_", 1)) if "_" in skew_name else (skew_name, None)  # type: ignore


def sample_skew_vs_base_pairs(
    skew_filepaths: dict[int, dict[str, list[str]]],
    base_filepaths: dict[int, dict[str, list[str]]],
    which: str,
    num_pairs: int,
    rng: np.random.Generator,
) -> list[tuple[str, str]]:
    """
    Sample pairs of trajectory files for KLD computation between skew and base systems.

    Args:
        skew_filepaths: Mapping from dimension to {skew_system: [filepaths]}.
        base_filepaths: Mapping from dimension to {base_system: [filepaths]}.
        which: One of "driver", "response", "base", "skew_intra", or "skew_inter".
            - "driver" or "response": pairs skew system with its driver/response base system.
            - "base": pairs skew system with a base system that is neither its driver nor response.
            - "skew_intra": pairs skew systems with same parents (intra-system).
            - "skew_inter": pairs skew systems with different parents (inter-system).
            - TODO: pair skew systems with skew systems that share the same driver XOR response
        num_pairs: Number of pairs to sample.
        rng: Numpy random number generator.

    Returns:
        List of (skew_filepath, base_filepath) tuples.
    """
    pairs: list[tuple[str, str]] = []
    if which == "skew_intra":
        # Only intra-system pairs (within the same skew system)
        intra_pairs: list[tuple[str, str]] = []
        for dim, skew_dim_dict in skew_filepaths.items():
            for skew_name, skew_files in skew_dim_dict.items():
                n = len(skew_files)
                if n >= 2:
                    all_pairs = [(skew_files[i], skew_files[j]) for i in range(n) for j in range(i + 1, n)]
                    intra_pairs.extend(all_pairs)
        if len(intra_pairs) > num_pairs:
            idxs = rng.choice(len(intra_pairs), num_pairs, replace=False)
            intra_pairs = [intra_pairs[i] for i in idxs]
        return intra_pairs
    elif which == "skew_inter":
        # Only inter-system pairs (between different skew systems, same dimension)
        inter_pairs: list[tuple[str, str]] = []
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
                    base_candidates = [name for name in base_dim_dict if name not in exclude and name is not None]
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


def base(
    num_base_systems: int,
    num_pairs: int,
    save_fname_suffix: str | None = None,
) -> None:
    """
    Compute and save KLD statistics for intra- and inter-system pairs among base systems.

    Args:
        num_base_systems: Number of base systems to sample.
        num_pairs: Number of pairs to sample for each pair type.
        save_fname_suffix: Optional suffix for output filename.
    """
    base_dir = os.path.join(DATA_DIR, base_split_name)
    base_system_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    sampled_base_systems = list(
        np.array(base_system_names)[
            rng.choice(
                len(base_system_names),
                size=min(num_base_systems, len(base_system_names)),
                replace=False,
            )
        ]
    )

    base_filepaths: dict[int, dict[str, list[str]]] = {}
    for system in sampled_base_systems:
        subdir = os.path.join(base_dir, system)
        for file in sorted(os.listdir(subdir)):
            filepath = os.path.join(subdir, file)
            coords, _ = load_trajectory_from_arrow(filepath)
            dim = coords.shape[0]
            base_filepaths.setdefault(dim, {}).setdefault(system, []).append(filepath)

    # Compute KLDs for intra- and inter-system pairs, store results in a dict
    base_kld_results: dict[str, dict[str, Any]] = {}

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
                "mean": float(np.mean(klds)),
                "std": float(np.std(klds)),
                "values": klds,
            }
        else:
            base_kld_results[pair_type] = {
                "pairs": pairs,
                "mean": None,
                "std": None,
                "values": [],
            }

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

    def convert_np(obj: Any) -> Any:
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


def skew(
    num_skew_systems: int,
    num_pairs: int,
    save_fname_suffix: str | None = None,
) -> None:
    """
    Compute and save KLD statistics for intra- and inter-system pairs among skew systems.

    Args:
        num_skew_systems: Number of skew systems to sample.
        num_pairs: Number of pairs to sample for each pair type.
        save_fname_suffix: Optional suffix for output filename.
    """
    skew_dir = os.path.join(DATA_DIR, skew_split_name)

    skew_system_names = [d for d in os.listdir(skew_dir) if os.path.isdir(os.path.join(skew_dir, d))]
    sampled_skew_systems = rng.choice(
        skew_system_names,
        size=min(num_skew_systems, len(skew_system_names)),
        replace=False,
    ).tolist()

    # Gather filepaths for base and skew systems with progress bars
    base_dir = os.path.join(DATA_DIR, base_split_name)
    base_system_names = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    base_filepaths = gather_filepaths_by_dim_and_system(base_dir, base_system_names, desc="Base systems")
    skew_filepaths = gather_filepaths_by_dim_and_system(skew_dir, sampled_skew_systems, desc="Skew systems")
    skew_kld_results: dict[str, dict[str, Any]] = {}

    for which in ["skew_intra", "skew_inter"]:
        if which == "skew_intra":
            print("Computing KLDs for skew-skew intra-system pairs...")
        elif which == "skew_inter":
            print("Computing KLDs for skew-skew inter-system pairs...")
        else:
            print(f"Computing KLDs for skew-{which} vs. base system pairs...")
        pairs = sample_skew_vs_base_pairs(skew_filepaths, base_filepaths, which, num_pairs, rng)
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
            "mean": float(np.mean(klds)) if klds else None,
            "std": float(np.std(klds)) if klds else None,
            "values": klds,
        }

    # Print concise summary
    for which, res in skew_kld_results.items():
        print(f"Skew-{which} vs. base system pairs: mean KLD={res['mean']}, std={res['std']}, n={len(res['values'])}")

        if save_fname_suffix is None:
            save_fname_suffix = ""
        output_json_path = os.path.join(
            "outputs/inheritance",
            f"{skew_split_name}_kld_results{save_fname_suffix}.json",
        )
        os.makedirs(os.path.dirname(output_json_path), exist_ok=True)

        def convert_np(obj: Any) -> Any:
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
    skew_split_name: str = "improved/final_skew40/train"
    base_split_name: str = "improved/final_base40/train"

    rseed: int = 987
    rng: np.random.Generator = np.random.default_rng(rseed)
    skew(num_skew_systems=1109, num_pairs=10000, save_fname_suffix=f"_rseed{rseed}")
