import hydra
import matplotlib.pyplot as plt
from dysts.flows import Lorenz, Rossler

from dystformer.skew_system import SkewProduct


def test_skew_system():
    sys = SkewProduct(
        driver=Rossler(),
        response=Lorenz(),
    )
    traj = sys.make_trajectory(
        8192,
        dt=min(sys.driver.metadata["dt"], sys.response.metadata["dt"]),
        pts_per_period=128,
        resample=True,
        standardize=False,
    )
    driver_traj = traj[:, : sys.driver_dim]
    response_traj = traj[:, sys.driver_dim :]
    print(driver_traj.shape, response_traj.shape)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"}
    )

    # Plot driver trajectory
    ax1.plot(driver_traj[:, 0], driver_traj[:, 1], driver_traj[:, 2])
    ax1.set_title("Driver (Lorenz)")

    # Plot response trajectory
    ax2.plot(response_traj[:, 0], response_traj[:, 1], response_traj[:, 2])
    ax2.set_title("Response (Rossler)")

    plt.tight_layout()
    plt.savefig("tests/figs/skew_system.png")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_skew_system()


if __name__ == "__main__":
    main()
