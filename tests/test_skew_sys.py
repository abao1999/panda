from dataclasses import field
from typing import Dict, Optional

import hydra
from dysts.flows import DynSys, Lorenz, ThomasLabyrinth


class SkewSystem(DynSys):
    def __init__(
        self,
        driver: DynSys,
        response: DynSys,
        dt: float = 0.01,
        period: Optional[float] = None,
        maximum_lyapunov_estimated: Optional[float] = None,
        **kwargs,
    ):
        super().__init__(
            metadata={**driver.metadata, **response.metadata},
            param_list=4,
            dt=dt,
            period=period,
            maximum_lyapunov_estimated=maximum_lyapunov_estimated,
            **kwargs,
        )
        self.coupling_map: Dict[int, int] = field(default_factory=additive_coupling_map)

    def rhs(self, t, X):
        pass


def test_skew_system():
    sys = SkewSystem(
        driver=Lorenz(),
        response=ThomasLabyrinth(),
    )


@hydra.main(config_path="../config", config_name="config", version_base=None)
def main(cfg):
    test_skew_system()


if __name__ == "__main__":
    main()
