"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""
import numpy as np

import dysts.flows as dfl

# from dysts.analysis import sample_initial_conditions
from scipy.integrate import solve_ivp

from typing import List, Union, Sequence
from chronos_dysts.utils import is_float_or_sequence_of_floats

import matplotlib.pyplot as plt
import os
import time
import argparse

from dyst_data import process_trajs


WORK_DIR = os.getenv('WORK', '')


def pad_array(arr: np.ndarray, n2: int, m2: int) -> np.ndarray:
    """
    General util to pad a 2D array to a target shape that is bigger than original shape
    """
    # Get the original shape (n1 x m1)
    n1, m1 = arr.shape
    # Calculate the padding amounts
    pad_rows = n2 - n1
    pad_cols = m2 - m1
    # Check if padding is needed
    if pad_rows < 0 or pad_cols < 0:
        raise ValueError("Target dimensions must be greater than or equal to original dimensions.")
    # Apply padding: ((before_rows, after_rows), (before_cols, after_cols))
    padded_arr = np.pad(arr, ((0, pad_rows), (0, pad_cols)), mode='constant', constant_values=0)
    return padded_arr


def construct_basic_affine_map(
        n: int, 
        m: int, 
        kappa: Union[float, Sequence[float]] = 1.0,
    ) -> np.ndarray:
    """
    Construct an affine map that sends (x, y, 1) -> (x, y, x + y)
    where x and y have lengths n and m respectively, and n <= m

    Parameters:
    n: driver system dimension
    m: response system dimension

    Returns:
    A: the affine map matrix (2D array), block matrix (n + 2m) x (n + m + 1)
    """
    # check if kappa is a float or a list of floats
    if isinstance(kappa, int): kappa = float(kappa)
    assert is_float_or_sequence_of_floats(kappa), "kappa must be a float or a list of floats"
    I_n = np.eye(n)  # n x n identity matrix
    I_m = np.eye(m)  # m x m identity matrix

    if isinstance(kappa, float):
        bottom_block = np.hstack([kappa * pad_array(I_n if n < m else I_m, m, n), I_m, np.zeros((m, 1))])
    else: # kappa is a list of floats
        k = min(n, m)
        assert len(kappa) == k, "coupling strength kappa must be of length min(n, m)" # type: ignore
        bottom_block = np.hstack([pad_array(np.diag(kappa), m, n), I_m, np.zeros((m, 1))])

    top_block = np.hstack([I_n, np.zeros((n, m)), np.zeros((n, 1)) ])
    middle_block = np.hstack([np.zeros((m, n)), I_m, np.zeros((m, 1))])
    
    A = np.vstack([top_block, middle_block, bottom_block])
    return A


def apply_affine_map(A: np.ndarray, x: np.ndarray, y: np.ndarray) -> List[np.ndarray]:
    """
    Apply an affine map to an augmented stacked vector of (x, y, 1)
    """
    n, m = len(x), len(y)
    # (x, y, 1) -> (x, y, x + y)
    transformed_vector = A @ np.hstack([x, y, 1])
    x = transformed_vector[:n]
    y = transformed_vector[n:n+m]
    return [x, y, transformed_vector[n+m:]]


def get_coupling(sys_driver, sys_response) -> np.ndarray:
    """
    Compute the coupling constants per dimension between the driver and response systems
    TODO: after making dyst_data, we have a folder of trajectories, we can use this to compute coupling strength by reading from arrow file instead of generating trajectories
    """
    print(f"Computing coupling strength between {sys_driver.name} and {sys_response.name}")
    n_driver, n_response = len(sys_driver.ic), len(sys_response.ic)
    k = min(n_driver, n_response)

    start_time = time.time()
    sol_driver = sys_driver.make_trajectory(1000)
    sol_response = sys_response.make_trajectory(1000)
    end_time = time.time()
    print(f"Trajectory generation time: {end_time - start_time} seconds")
    amp_driver = np.mean(np.abs(sol_driver), axis=0)
    amp_response = np.mean(np.abs(sol_response), axis=0)
    print("Amplitude of driver system: ", amp_driver)
    print("Amplitude of response system: ", amp_response)

    kappa =  amp_response[:k] / amp_driver[:k]

    print("Coupling strength: ", kappa)
    return kappa


def skew_sys(sys_driver, sys_response, affine_map):
    """
    Wrapper for skew-product system rhs, taking in a pre-computed affine map
    """
    n_driver, n_response = len(sys_driver.ic), len(sys_response.ic)
    def _skew_rhs(t, combined_ics):
        """
        Skew-product system rhs signature
        """
        # Split the combined initial conditions into driver and response systems
        x_driver, x_response = combined_ics[:n_driver], combined_ics[n_driver: n_driver + n_response] 
        
        # # Method 1
        # _, _, x_response = apply_affine_map(affine_map, x_driver, x_response)
        # # assert len(x_response) == n_response

        # Compute the flow of the driver and response systems
        flow_driver = np.array(sys_driver.rhs(x_driver, t))
        flow_response = np.array(sys_response.rhs(x_response, t))

        # # Method 2 - this seems like a weird thing to do
        # flow_driver, _, flow_response = apply_affine_map(affine_map, flow_driver, flow_response)

        # Method 3 - normalize flow rhs on the fly, both to unit norm
        # kappa = np.linalg.norm(flow_response) / np.linalg.norm(flow_driver)
        kappa_driver = 1 / np.linalg.norm(flow_driver)
        kappa_response = 1 / np.linalg.norm(flow_response)
        # affine_map = construct_basic_affine_map(n_driver, n_response, kappa)
        flow_response = (kappa_driver * flow_driver) + (kappa_response * flow_response)

        # kappa = np.linalg.norm(flow_response) / np.linalg.norm(flow_driver)
        # print("kappa: ", kappa)
        # # affine_map = construct_basic_affine_map(n_driver, n_response, kappa)
        # flow_response = 0.5 * (kappa * flow_driver) + 0.5 * (flow_response)

        # kappa = 1 / np.linalg.norm(flow_response + flow_driver)
        # print("kappa: ", kappa)
        # # affine_map = construct_basic_affine_map(n_driver, n_response, kappa)
        # flow_response = kappa * (flow_driver + flow_response)

        skew_flow = np.concatenate([flow_driver, flow_response])
        return skew_flow

    return _skew_rhs


def run_skew_product_system(sys_driver, sys_response):
    """
    Run the skew-product system and return the trajectory of the driver and response systems
    """
    n_driver, n_response = len(sys_driver.ic), len(sys_response.ic)
    print(f"dimension of driver system: {n_driver}, dimension of response system: {n_response}")
    # # Sample a random initial condition for the response system
    # ic = sample_initial_conditions(sys_response, 1)[0] # number of response systems is 1.. this is super slow, don't use this
    # # TODO: Check if the following line is correct
    # sys_response.ic = ic

    # combine initial conditions of driver and response system
    combined_ics = np.concatenate([np.array(sys_driver.ic), np.array(sys_response.ic)])
    # get integration arguments from comparing the driver and response systems
    dt = min(sys_driver.dt, sys_response.dt)
    tlim = 20 * max(sys_driver.period, sys_response.period)
    tpts = np.linspace(0, tlim, 10_000)
    
    # # construct affine maps
    # kappa = get_coupling(sys_driver, sys_response)
    kappa = np.ones(n_response) # dummy to avoid computing coupling strength
    basic_affine_map = construct_basic_affine_map(n_driver, n_response, kappa)
    print("Affine map:")
    print(basic_affine_map)

    # set up skew system rhs and solve
    skew_rhs = skew_sys(
        sys_driver, 
        sys_response, 
        basic_affine_map,
    )
    start_time = time.time()
    sol = solve_ivp(
            skew_rhs, 
            [0, tlim],
            combined_ics, 
            t_eval=tpts, 
            first_step=dt,
            method="Radau",
            rtol=1e-6,
            atol=1e-6,
        )
    end_time = time.time()
    print(f"Integration time: {end_time - start_time} seconds")

    sol_driver = sol.y[:n_driver]
    sol_response = sol.y[n_driver: n_driver + n_response]
    return [sol_driver, sol_response]


def test_map():
    """
    Simple test for affine map construction and application
    """
    x = np.array([1, 2, 3])
    y = np.array([4, 5, 6, 7])
    kappa = [1, 2, 3]
    A = construct_basic_affine_map(len(x), len(y), kappa=kappa)
    print(A)
    res = apply_affine_map(A, x, y)
    print(res)
    print(np.concatenate(res))


if __name__ == "__main__":
    # test_map()

    parser = argparse.ArgumentParser()
    parser.add_argument("dysts_names", help="Names of the dynamical systems", nargs="+", type=str)
    args = parser.parse_args()
    dysts_names = args.dysts_names
    
    assert len(dysts_names) == 2, "Must provide exactly two dynamical system names for now (TODO: generalize to n systems)"
    driver_name, response_name = dysts_names
    driver_sys = getattr(dfl, driver_name)()
    response_sys = getattr(dfl, response_name)()

    # get_coupling(driver_sys, response_sys)

    # TODO: wrap this in loops to sample initial conditions and parameter perturbations
    res = run_skew_product_system(driver_sys, response_sys)
    print(res)
    sol_response = res[1]
    print("response solution shape: ", sol_response.shape)


    save_dir = "figs"
    os.makedirs(save_dir, exist_ok=True)
    plot_name = f"{driver_name}_driving_{response_name}"
    # 3D plot (first three coordinates)
    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    print("Plotting 3D trajectories and saving to ", save_path)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection='3d')
    # plot x and y and z
    ax.plot(sol_response[0, :], sol_response[1, :], sol_response[2, :], alpha=0.8, linewidth=1)  # X,Y,Z
    ax.scatter(*sol_response[:3, 0], marker="*", s=100, alpha=0.5)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
    ax.ticklabel_format(style='sci', scilimits=(0,0), axis='both')
    plt.title(plot_name.replace('_', ' '))
    plt.savefig(save_path, dpi=300)
    plt.close()

    traj_save_path = os.path.join(WORK_DIR, "data", "skew_systems")
    print("Saving trajectories to ", traj_save_path)

    skew_name = f"{driver_name}_{response_name}"
    skew_dict = {skew_name: np.expand_dims(sol_response, axis=0)} # we have only one sample (ic + param perturbation)
    process_trajs(traj_save_path, skew_dict, verbose=True)