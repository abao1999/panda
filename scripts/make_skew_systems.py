"""
Search for valid skew-product dynamical sytems and generate trajectory datasets
"""

import argparse
import os

import dysts.flows as dfl
import matplotlib.pyplot as plt

from dystformer.skew_system import SkewSystem

WORK_DIR = os.getenv("WORK", "")

if __name__ == "__main__":
    # test_map()

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "dysts_names", help="Names of the dynamical systems", nargs="+", type=str
    )
    parser.add_argument(
        "--compute_coupling_strength",
        help="Whether to compute coupling strength",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--couple_phase_space",
        help="Whether to couple phase space",
        type=bool,
        default=True,
    )
    parser.add_argument(
        "--couple_flows",
        help="Whether to couple flows",
        type=bool,
        default=False,  # can do both types of coupling
    )
    args = parser.parse_args()
    dysts_names = args.dysts_names

    assert (
        len(dysts_names) == 2
    ), "Must provide exactly two dynamical system names for now (TODO: generalize to n systems)"

    driver_name, response_name = dysts_names
    driver_sys = getattr(dfl, driver_name)()
    response_sys = getattr(dfl, response_name)()

    skew_system = SkewSystem(
        driver_sys,
        response_sys,
        compute_coupling_strength=args.compute_coupling_strength,
    )
    skew_sol = skew_system.run(
        couple_phase_space=args.couple_phase_space, couple_flows=args.couple_flows
    )
    sol_response = skew_sol[1]
    print("response solution shape: ", sol_response.shape)

    save_dir = "figs"
    os.makedirs(save_dir, exist_ok=True)
    plot_name = f"{driver_name}_driving_{response_name}"
    # 3D plot (first three coordinates)
    save_path = os.path.join(save_dir, f"{plot_name}_3D.png")
    print("Plotting 3D trajectories and saving to ", save_path)
    fig = plt.figure(figsize=(6, 6))
    ax = fig.add_subplot(111, projection="3d")
    # plot x and y and z
    ax.plot(
        sol_response[0, :],
        sol_response[1, :],
        sol_response[2, :],
        alpha=0.8,
        linewidth=1,
    )  # X,Y,Z
    ax.scatter(*sol_response[:3, 0], marker="*", s=100, alpha=0.5)
    ax.set_xlabel("X")
    ax.set_ylabel("Y")
    ax.set_zlabel("Z")  # type: ignore
    ax.tick_params(pad=3)  # Increase the padding between ticks and axes labels
    ax.ticklabel_format(style="sci", scilimits=(0, 0), axis="both")
    plt.title(plot_name.replace("_", " "))
    plt.savefig(save_path, dpi=300)
    plt.close()

    # traj_save_path = os.path.join(WORK_DIR, "data", "skew_systems")
    # print("Saving trajectories to ", traj_save_path)

    # skew_name = f"{driver_name}_{response_name}"
    # skew_dict = {
    #     skew_name: np.expand_dims(sol_response, axis=0)
    # }  # we have only one sample (ic + param perturbation)
    # process_trajs(traj_save_path, skew_dict, verbose=True)
