import argparse
import json
import logging
import os
from typing import Any, Callable

import dysts.flows as flows
import numpy as np
from dysts.systems import DynSys
from scipy.optimize import approx_fprime

from dystformer.coupling_maps import RandomAdditiveCouplingMap
from dystformer.skew_system import SkewProduct
from dystformer.utils import plot_trajs_multivariate

WORK_DIR = os.getenv("WORK", "")
DATA_DIR = os.path.join(WORK_DIR, "data")
PARAMS_DIR = os.path.join(DATA_DIR, "parameters")

logger = logging.getLogger(__name__)


def compute_jac_fd(
    rhs: Callable[[np.ndarray, float], np.ndarray],
    y_val: np.ndarray,
    t_val: float,
    eps: float = 1e-8,
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
    param_dict: dict[str, Any],
    **kwargs,
) -> DynSys:
    """
    Initialize a skew-product dynamical system from saved parameters.
    Assumes RandomAdditiveCouplingMap.
    """
    system_name = f"{driver_name}_{response_name}"
    required_keys = [
        "driver_params",
        "response_params",
        "coupling_map",
    ]
    for key in required_keys:
        if key not in param_dict:
            raise ValueError(f"Key {key} not found in param_dict for {system_name}")

    driver = getattr(flows, driver_name)(parameters=param_dict["driver_params"])
    response = getattr(flows, response_name)(parameters=param_dict["response_params"])

    coupling_map = RandomAdditiveCouplingMap._deserialize(param_dict["coupling_map"])

    sys = SkewProduct(
        driver=driver, response=response, coupling_map=coupling_map, **kwargs
    )
    # logger.info(f"param dict for {system_name}: \n {param_dict}")
    # logger.info(f"coupling map for {system_name}: \n {coupling_map}")

    return sys


def test_system_jacobian(
    system_name: str,
    param_dict: dict[str, Any],
    num_timesteps: int = 4096,
    num_periods: int = 10,
    transient: int = 200,
    n_points_sample: int = 10,
    eps: float = 1e-8,
    save_traj_plot_dir: str | None = None,
    verbose: bool = False,
) -> int:
    """
    Test the Jacobian of a system by comparing the analytic and finite difference
    Returns: 0 if failed, 1 if passed, 2 if not analytic Jacobian not implemented
    """
    is_skew = "_" in system_name
    if is_skew:
        # hack to get driver and response names from system name
        driver_name, response_name = system_name.split("_")
        sys = init_skew_system_from_params(driver_name, response_name, param_dict)

    else:
        params = param_dict["params"]
        sys = getattr(flows, system_name)(parameters=params)

        assert sys.dimension == param_dict["dim"], "Dimension mismatch!"

    # set initial condition
    sys.ic = np.array(param_dict["ic"])

    if not sys.has_jacobian():
        return 2

    ts, traj = sys.make_trajectory(
        num_timesteps,
        pts_per_period=num_timesteps // num_periods,
        return_times=True,
        atol=1e-10,
        rtol=1e-9,
    )
    if save_traj_plot_dir is not None:
        traj_to_plot = traj[None, :, :].transpose(0, 2, 1)
        driver_coords = traj_to_plot[:, : sys.driver_dim, :]
        response_coords = traj_to_plot[:, sys.driver_dim :, :]
        for name, coords in [
            ("driver", driver_coords),
            ("response", response_coords),
        ]:
            plot_trajs_multivariate(
                coords,
                save_dir=save_traj_plot_dir,
                plot_name=f"reconstructed_{system_name}_{name}",
                standardize=True,
                plot_projections=True,
            )

    assert traj is not None, f"{system_name} should be integrable"
    ts, traj = ts[transient:], traj[transient:]
    # sample n_points_sample points from the trajectory
    sample_indices = np.random.choice(len(traj), size=n_points_sample, replace=False)
    traj_sample = traj[sample_indices]
    ts_sample = ts[sample_indices]

    for y_val, t_val in zip(traj_sample, ts_sample):
        if verbose:
            logger.info(f"Testing analytic versus fd jac at t={t_val}, y={y_val}")
        jac_fd = compute_jac_fd(sys.rhs, y_val, t_val, eps)
        jac_analytic = sys.jac(y_val, t_val)
        if not np.allclose(jac_fd, jac_analytic, atol=1e-5):
            return 0
    return 1


if __name__ == "__main__":
    logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--params_dir",
        help="Directory to save parameters",
        type=str,
        default=PARAMS_DIR,
    )
    parser.add_argument(
        "--split", help="Split of the data", type=str, default="debug_base_train"
    )
    parser.add_argument(
        "--n_systems", help="Number of systems to test", type=int, default=1
    )
    parser.add_argument(
        "--eps",
        help="Step size for the finite difference calculation",
        type=float,
        default=1e-8,
    )
    parser.add_argument(
        "--plot_save_dir",
        help="Directory to save trajectory plots",
        type=str,
        default="tests/figs",
    )
    args = parser.parse_args()

    if args.split is None:
        raise ValueError("Split must be provided")

    params_json_path = os.path.join(args.params_dir, args.split, "successes.json")
    with open(params_json_path, "r") as f:
        # dict of dicts of params for each system
        all_param_dicts = json.load(f)

    if args.plot_save_dir is not None:
        os.makedirs(args.plot_save_dir, exist_ok=True)

    for i, (system, param_dicts_all_samples) in enumerate(all_param_dicts.items()):
        if i >= args.n_systems:
            break
        # NOTE: for now we only test the first sample for each system
        param_dict = param_dicts_all_samples[0]
        status = test_system_jacobian(
            system,
            param_dict,
            eps=args.eps,
            save_traj_plot_dir=args.plot_save_dir,
        )
        if status == 0:
            logger.info(f"FAILED for {system}")
        elif status == 1:
            logger.info(f"PASSED for {system}")
        elif status == 2:
            logger.info(f"NOT IMPLEMENTED for {system}")
