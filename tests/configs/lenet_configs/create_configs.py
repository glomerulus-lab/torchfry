"""
Generate JSON configuration files for experiments using FastfoodLayer and RKSLayer
with various combinations of learnable parameters. Files are saved in the current directory.
"""

import os
import json
from itertools import product

# Ensure the output directory exists
output_dir = os.path.join(os.getcwd(), "tests/configs/lenet_configs")
os.makedirs(output_dir, exist_ok=True)

# Final combined JSON output
combined_config = {}

# Base shared configuration
base_config = {
    "scale": 0.1,
    "lr": 0.001,
    "mb": 512,
    "epochs": 200,
    "trials": 10,
    "widths": [8192, 8192, 8192],
    "nonlinearity": False,
    "hadamard": "Dao",
}

# Generate FastfoodLayer configurations. All combinations of learnable.
for learn_S, learn_G, learn_B in product([False, True], repeat=3):
    config = {
        "layer": "FastfoodLayer",
        "learn_S": learn_S,
        "learn_G": learn_G,
        "learn_B": learn_B
    }
    config.update(base_config)
    key = f"Fastfood_S{int(learn_S)}_G{int(learn_G)}_B{int(learn_B)}"
    combined_config[key] = config

# Generate RKSLayer configurations
for learn_G in [False, True]:
    config = {
        "layer": "RKSLayer",
        "learn_G": learn_G
    }
    config.update({k: v for k, v in base_config.items() if k != "hadamard"})
    key = f"RKS_G{int(learn_G)}"
    combined_config[key] = config

# Write the combined config to a single JSON file
output_path = os.path.join(output_dir, "config.json")
with open(output_path, "w", encoding="utf-8") as f:
    json.dump(combined_config, f, indent=2)

print(f"All configuration files written to {output_path}")
