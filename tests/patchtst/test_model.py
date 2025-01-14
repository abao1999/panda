import hydra
import torch

from dystformer.patchtst.model import PatchTST


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    model = PatchTST(dict(cfg.patchtst), mode="predict", device="cuda")
    model.eval()

    x = torch.randn(4, 512, 3).to("cuda")

    for i in range(10):
        outputs = model(x)
        x = torch.cat([x, outputs.prediction_outputs], dim=1)

    assert x.shape == torch.Size([4, 512 + 64 * 10, 3])
    breakpoint()


if __name__ == "__main__":
    main()
