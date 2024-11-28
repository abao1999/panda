import os

from safetensors import safe_open
from safetensors.torch import load_file, save_file

WORK_DIR = os.getenv("WORK", "")
CHECKPOINT_DIR = os.path.join(WORK_DIR, "checkpoints")

checkpoint_path = os.path.join(CHECKPOINT_DIR, "run-303", "checkpoint-final")

# Path to your input safetensors file
input_path = os.path.join(checkpoint_path, "old_model.safetensors")
# Path to save the modified file
output_path = os.path.join(checkpoint_path, "model.safetensors")

# Load the contents of the safetensors file
tensors = load_file(input_path)

# Open the safetensors file to read metadata
with safe_open(input_path, framework="pt") as f:
    metadata = f.metadata()

if metadata is None:
    metadata = {"format": "pt"}

# Update the keys in the tensors
updated_tensors = {key[len("model.") :]: value for key, value in tensors.items()}

print(updated_tensors.keys())

# Save the updated tensors to a new safetensors file with the original metadata
save_file(updated_tensors, output_path, metadata=metadata)

print(f"Updated keys saved to {output_path}")
