import hydra
import matplotlib.pyplot as plt
from dysts.flows import CircadianRhythm, ExcitableCell

from dystformer.skew_system import SkewProduct


def test_skew_system():
    driver = ExcitableCell()
    response = CircadianRhythm()
    driver, response = response, driver
    sys = SkewProduct(
        driver=driver,
        response=response,
        _default_random_seed=None,
    )
    traj = sys.make_trajectory(
        1024,
        pts_per_period=1024 // 5,
        timescale="Fourier",
        resample=True,
        standardize=False,
    )

    if traj is None:
        print("Integration failed")
        exit()

    driver_traj = traj[:, : sys.driver_dim]
    response_traj = traj[:, sys.driver_dim :]
    print(driver_traj.shape, response_traj.shape)

    fig, (ax1, ax2) = plt.subplots(
        1, 2, figsize=(12, 5), subplot_kw={"projection": "3d"}
    )

    # Plot driver trajectory
    ax1.plot(driver_traj[:, 0], driver_traj[:, 1], driver_traj[:, 2])
    ax1.set_title(f"Driver ({sys.driver.name})")

    # Plot response trajectory
    ax2.plot(response_traj[:, 0], response_traj[:, 1], response_traj[:, 2])
    ax2.set_title(f"Response ({sys.response.name})")

    plt.tight_layout()
    plt.savefig("tests/figs/skew_system.png")


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_skew_system()


if __name__ == "__main__":
    main()
