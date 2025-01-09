import argparse
import json
import os
from typing import Callable, List

import dysts.flows as flows
import numpy as np
from dysts.systems import DynSys
from scipy.optimize import approx_fprime

from dystformer.skew_system import SkewProduct

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")
PARAMS_DIR = os.path.join(DATA_DIR, "parameters")


def compute_jac_fd(
    rhs: Callable,
    y_val: np.ndarray,
    t_val: float,
    eps: float = 1e-3,
) -> np.ndarray:
    """Calculate numerical jacobian of a function with respect to a reference value"""
    func = lambda x: np.array(rhs(x, t_val))
    y_val = np.array(y_val)

    d = len(y_val)
    all_rows = list()
    for i in range(d):
        row_func = lambda yy: func(yy)[i]
        row = approx_fprime(y_val, row_func, epsilon=eps)
        all_rows.append(row)
    jac = np.array(all_rows)

    return jac


def init_skew_system_from_params(
    driver_name: str,
    response_name: str,
    params: dict[str, float],
    **kwargs,
) -> DynSys:
    """
    Initialize a skew-product dynamical system from saved parameters. Rough template for now.
    Need to implement some additional param saving to make this work.
    """
    driver = getattr(flows, driver_name)()
    response = getattr(flows, response_name)()

    # TODO: get driver_scale and response_scale from saved params
    # TODO: get entire coupling map from saved params
    coupling_map = None

    return SkewProduct(
        driver=driver, response=response, coupling_map=coupling_map, **kwargs
    )


def test_system_jacobian(
    system_name: str,
    params: List[float] | dict[str, float],
    num_timesteps: int = 4096,
    num_periods: int = 10,
    transient: int = 200,
    n_points_sample: int = 10,
    eps: float = 1e-8,
) -> bool:
    is_skew = "_" in system_name
    if is_skew:
        # hack to get driver and response names from system name
        driver_name, response_name = system_name.split("_")
        # params should include driver_scale, response_scale, and coupling_map
        sys = init_skew_system_from_params(driver_name, response_name, params)

    else:
        sys = getattr(flows, system_name)(params)

    print(f"System: {system_name}, params: {sys.params}, ic: {sys.ic}")
    if not sys.has_jacobian():
        print(f"No analytic Jacobian implemented for {system_name}")
        return False

    print(f"Testing {system_name}")
    ts, traj = sys.make_trajectory(
        num_timesteps,
        pts_per_period=num_timesteps // num_periods,
        return_times=True,
        atol=1e-10,
        rtol=1e-9,
    )
    print(f"Trajectory shape: {traj.shape}")
    assert traj is not None, f"{system_name} should be integrable"
    ts, traj = ts[transient:], traj[transient:]
    # sample n_points_sample points from the trajectory
    sample_indices = np.random.choice(len(traj), size=n_points_sample, replace=False)
    traj_sample = traj[sample_indices]
    ts_sample = ts[sample_indices]
    print(f"Sample indices: {sample_indices}")
    print(f"Sample trajectory shape: {traj_sample.shape}")
    print(f"Sample time shape: {ts_sample.shape}")

    for y_val, t_val in zip(traj_sample, ts_sample):
        print(f"Testing {system_name} at t={t_val}, y={y_val}")
        jac_fd = compute_jac_fd(sys.rhs, y_val, t_val, eps)
        jac_analytic = sys.jac(y_val, t_val)
        assert np.allclose(
            jac_fd, jac_analytic, atol=1e-5
        ), f"Jacobian for {system_name} is incorrect"
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params_dir",
        help="Directory to save parameters",
        type=str,
        default=PARAMS_DIR,
    )
    parser.add_argument("--split", help="Split of the data", type=str, default="train")
    parser.add_argument(
        "--n_systems", help="Number of systems to test", type=int, default=1
    )
    parser.add_argument(
        "--eps",
        help="Step size for the finite difference calculation",
        type=float,
        default=1e-8,
    )
    args = parser.parse_args()

    if args.split is None:
        raise ValueError("Split must be provided")

    params_json_path = os.path.join(args.params_dir, args.split, "successes.json")
    with open(params_json_path, "r") as f:
        all_params_dict = json.load(f)

    for i, (system, params_all_samples) in enumerate(all_params_dict.items()):
        # NOTE: right now we only test the first sample for each system
        params = params_all_samples[0]
        if i >= args.n_systems:
            break
        test_system_jacobian(system, params, eps=args.eps)
