import os
import numpy as np
from functools import partial
from dataclasses import dataclass
from typing import List, Optional, Callable, Dict
from tqdm import trange

from dysts.base import make_trajectory_ensemble, init_cond_sampler
from chronos_dysts.attractor import (
    EnsembleCallbackHandler,
    check_no_nans,
    check_boundedness,
    check_not_fixed_point,
    check_not_limit_cycle,
    check_power_spectrum,
    check_stationarity,
)
from chronos_dysts.utils import (
    split_systems,
    process_trajs,
    filter_dict,
)


WORK_DIR = os.getenv('WORK')
DELAY_SYSTEMS = ['MackeyGlass', 'IkedaDelay', 'SprottDelay', 'VossDelay', 'ScrollDelay', 'PiecewiseCircuit']
FIGS_SAVE_DIR = "tests/figs"


@dataclass
class ParamPerturb:
    scale: float
    random_seed: int = 0

    def __post_init__(self) -> None:
        self.rng = np.random.default_rng(self.random_seed)

    def __call__(self, name: str, param: np.ndarray) -> np.ndarray:
        print("param name: ", name)
        print("Original param: ", param)
        # perturbed = np.random.normal(
        perturbed = self.rng.normal(
            loc=param,
            scale=self.scale,
        )
        print("Perturbed param: ", perturbed)
        return perturbed

# Event function to check if integration is taking too long
import time
class TimeLimitEvent:
    def __init__(self, max_duration):
        self.start_time = None
        self.max_duration = max_duration

    def __call__(self, t, y):
        if self.start_time is None:
            self.start_time = time.time()
        elapsed_time = time.time() - self.start_time
        if elapsed_time > self.max_duration:
            print("Integration stopped due to time limit.")
            return 0  # Trigger the event
        return 1  # Continue the integration

# Event function to detect instability
def instability_event(t, y):
    # Example criterion: If the solution's magnitude exceeds a large threshold
    if np.any(np.abs(y) > 1e4): # reasonable threshold, since we are standardizing trajectories
        print("y: ", y)
        print("Integration stopped due to instability.")
        return 0  # Trigger the event
    return 1  # Continue the integration


def save_dyst_ensemble(
    dysts_names: List[str] = ['Lorenz'],
    split: str = "train",
    rseed: int = 999,
    num_periods: int = 5,
    num_points: int = 1024,
    num_ics: int = 3,
    num_param_perturbations: int = 1,
    samples_save_interval: int = 1,
    events: Optional[List[Callable]] = None,
    callback_handler: Optional[Callable] = None,
) -> None:
    print(f"Making {split} split with {len(dysts_names)} dynamical systems: \n {dysts_names}")
    
    ic_sampler = init_cond_sampler(subset=dysts_names, random_seed=rseed)
    param_sampler = ParamPerturb(scale=1, random_seed=rseed)

    num_total_samples = num_param_perturbations * num_ics
    ensemble_list = []

    for i in range(num_param_perturbations):
        for j in trange(num_ics):
            sample_idx = i + j

            print("Making ensemble for sample ", sample_idx)
            # each ensemble is of type Dict[str, ndarray]
            ensemble = make_trajectory_ensemble(
                num_points, subset=dysts_names, use_multiprocessing=True, 
                init_conds=ic_sampler(scale=1) if num_ics > 1 else {}, 
                param_transform=param_sampler, #if num_param_perturbations > 1 else None,
                use_tqdm=True, standardize=False, pts_per_period=num_points//num_periods,
                events=events,
            )
            ensemble, excluded_keys = filter_dict(ensemble) #, req_num_vals=num_points)
            print("INTEGRATION FAILED FOR:", excluded_keys)

            ensemble_list.append(ensemble)

            # save samples of trajectory ensembles to arrow files and clear list of ensembles
            # Essentially a batched version of process_trajs
            if ((sample_idx + 1) % samples_save_interval) == 0 or (sample_idx + 1) == num_total_samples:
                # transpose and stack to get shape (num_samples, num_dims, num_timesteps) from original (num_timesteps, num_dims)
                ensemble_keys = set().union(*[d.keys() for d in ensemble_list])
                ensemble = {key: np.stack([d[key] for d in ensemble_list if key in d], axis=0).transpose(0, 2, 1) for key in ensemble_keys}

                # TODO: callbacks
                print("Checking ensemble for attractor properties")
                if callback_handler:
                    callback_handler.plot_phase_space(ensemble, save_dir=FIGS_SAVE_DIR)
                    callback_handler.execute_callbacks(ensemble, first_sample_idx=sample_idx + 1 - samples_save_interval) # first index of current batch of samples
                    callback_handler.check_status_all()

                # print(f"Saving {len(ensemble_list)} sampled trajectoies to arrow files")
                # data_dir = os.path.join(WORK_DIR, f'data/{split}')
                # os.makedirs(data_dir, exist_ok=True)
                # process_trajs(data_dir, ensemble, verbose=True)
                # reset lists of ensembles
                ensemble_list = []

    if callback_handler:
        callback_handler.check_status_all()

if __name__ == '__main__':

    # set random seed
    rseed = 999 # we are using same seed for split and ic and param samplers

    # generate split of dynamical systems
    # _, train = split_systems(0.3, seed=rseed, excluded_systems=DELAY_SYSTEMS) # + FORCED_SYSTEMS)
    train = ['Tsucs2']
    # events for solve_ivp
    time_limit_event = TimeLimitEvent(max_duration=60)  # 1 min time limit
    time_limit_event.terminal = True
    instability_event.terminal = True

    print("Setting up callbacks for attractor properties")
    # callbacks to check attractor validity when creating traj ensemble of dysts
    ens_callback_handler = EnsembleCallbackHandler(verbose=2)
    ens_callback_handler.add_callback(check_no_nans)
    ens_callback_handler.add_callback(check_boundedness)
    ens_callback_handler.add_callback(check_not_fixed_point)
    ens_callback_handler.add_callback(
        partial(
            check_not_limit_cycle, 
            threshold=1e-3, 
            plot_save_dir=FIGS_SAVE_DIR # NOTE: set to None when actually generating data so we don't plot thousands of times
        )
    )
    ens_callback_handler.add_callback(
        partial(
            check_power_spectrum, 
            timestep=1.0, # NOTE: expose self.dt from dysts
            plot_save_dir=FIGS_SAVE_DIR # NOTE: set to None when actually generating data so we don't plot thousands of times
        )
    )
    ens_callback_handler.add_callback(check_stationarity)

    # make the train split
    save_dyst_ensemble(
        train,
        split="train", 
        rseed=rseed,
        num_periods=5,
        num_points=1024,
        num_ics=1,
        num_param_perturbations=1,
        events=[time_limit_event, instability_event],
        callback_handler=ens_callback_handler,
    )
    

# # Function to compute Poincaré section
# def poincare_section(x, y, z, section_plane='xy', section_value=0, tolerance=1e-3):
#     """
#     Analyze and visualize the Poincaré section with tolerance.

#     Args:
#         x, y, z (array): State variables.
#         section_plane (str): Plane for Poincaré section ('xy', 'xz', 'yz').
#         section_value (float): Value of the plane where section is taken.
#         tolerance (float): Tolerance for detecting the plane crossing.

#     Returns:
#         ndarray: Array of indices where the Poincaré section occurs.
#     """
#     indices = []
#     for i in range(1, len(x)):
#         if section_plane == 'xy':
#             if abs(z[i] - section_value) < tolerance:
#                 indices.append(i)
#         elif section_plane == 'xz':
#             if abs(y[i] - section_value) < tolerance:
#                 indices.append(i)
#         elif section_plane == 'yz':
#             if abs(x[i] - section_value) < tolerance:
#                 indices.append(i)
#         else:
#             raise ValueError("Invalid section plane. Choose from 'xy', 'xz', 'yz'.")

#     # Extract points near the Poincaré section
#     # TODO: generalize to other section planes
#     x_section = np.array([x[i] for i in indices])
#     y_section = np.array([y[i] for i in indices])
    
#     return x_section, y_section

# # Compute and plot Poincaré section
# def plot_poincare_section(x_section, y_section, section_plane):
#     """
#     Plot the Poincaré section.

#     Args:
#         x_section, y_section (array): Points on the Poincaré section.
#         section_plane (str): Plane for Poincaré section ('xy', 'xz', 'yz').
#     """
#     plt.figure(figsize=(8, 6))
#     plt.scatter(x_section, y_section, color='red', s=10)
#     plt.title(f'Poincaré Section on {section_plane}-plane')
#     plt.xlabel(section_plane[0].upper())
#     plt.ylabel(section_plane[1].upper())
#     plt.grid(True)
#     plt.savefig(os.path.join(FIGS_SAVE_DIR, "poincare_section.png"), dpi=300)

# # Parameters for Poincaré section analysis
# section_plane = 'xy'
# section_value = 20  # Plane value for the section
# tolerance = 1e-1   # Tolerance for detecting the plane crossing

# # Get Poincaré section points
# x_section, y_section = poincare_section(x, y, z, section_plane, section_value, tolerance)

# # Plot the Poincaré section
# plot_poincare_section(x_section, y_section, section_plane)




# # Optionally, you can analyze the periodicity of the Poincaré section points
# def analyze_periodicity(x_section, y_section):
#     """
#     Analyze periodicity of the Poincaré section points.
    
#     Args:
#         x_section, y_section (array): Points on the Poincaré section.

#     Returns:
#         None: Displays plots for analysis.
#     """
#     plt.figure(figsize=(12, 6))
#     plt.subplot(1, 2, 1)
#     plt.plot(x_section, y_section, 'o')
#     plt.title('Poincaré Section Points')
#     plt.xlabel('X')
#     plt.ylabel('Y')
#     plt.grid(True)

#     plt.subplot(1, 2, 2)
#     plt.hist(np.sqrt(np.diff(x_section)**2 + np.diff(y_section)**2), bins=50)
#     plt.title('Histogram of Distances Between Successive Points')
#     plt.xlabel('Distance')
#     plt.ylabel('Frequency')
#     plt.grid(True)
#     plt.savefig(os.path.join(FIGS_SAVE_DIR, "poincare_distances.png"), dpi=300)

#     # Simple heuristic check for periodic behavior
#     if len(x_section) > 10:
#         distances = np.sqrt(np.diff(x_section)**2 + np.diff(y_section)**2)
#         mean_distance = np.mean(distances)
#         std_distance = np.std(distances)
#         print(f"Mean distance between successive points: {mean_distance:.4f}")
#         print(f"Standard deviation of distances: {std_distance:.4f}")

#         if std_distance < tolerance:
#             print("The Poincaré section suggests a periodic behavior (limit cycle).")
#         else:
#             print("The Poincaré section does not show clear periodic behavior.")

# # Optionally analyze the periodicity of the Poincaré section points
# analyze_periodicity(x_section, y_section)
