import hydra
import torch
from huggingface_hub import hf_hub_download
from transformers import PatchTSTConfig, PatchTSTForPrediction


@hydra.main(config_path="../config", config_name="model", version_base=None)
def main(cfg):
    print(cfg.patchtst_params)
    # Initializing an PatchTST configuration with 12 time steps for prediction
    configuration = PatchTSTConfig(**dict(cfg.patchtst_params))

    # Randomly initializing a model (with random weights) from the configuration
    model = PatchTSTForPrediction(configuration)

    # Accessing the model configuration
    configuration = model.config

    file = hf_hub_download(
        repo_id="hf-internal-testing/etth1-hourly-batch",
        filename="train-batch.pt",
        repo_type="dataset",
    )
    batch = torch.load(file)
    print(batch["past_values"].shape)
    print(batch["past_values"][0])
    print(batch["future_values"].shape)

    # during training, one provides both past and future values
    outputs = model(
        past_values=batch["past_values"],
        future_values=batch["future_values"],
    )

    print(list(outputs.keys()))
    print(outputs.loc.shape)
    print(outputs.scale.shape)
    print(outputs.prediction_outputs.shape)
    print(outputs.loss)


if __name__ == "__main__":
    main()
