import hydra
import torch

from dystformer.chronos.pipeline import ChronosPipeline


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg):
    torch_dtype = getattr(torch, cfg.eval.torch_dtype)
    pipeline = ChronosPipeline.from_pretrained(
        cfg.eval.checkpoint_path,
        device_map=cfg.eval.device,
        torch_dtype=torch_dtype,
    )
    breakpoint()


if __name__ == "__main__":
    main()
