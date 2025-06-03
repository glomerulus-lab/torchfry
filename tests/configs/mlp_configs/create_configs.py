"""
Generate JSON configuration files for FastfoodLayer and RKSLayer experiments with
various learnable parameter combinations. Files are saved in the current directory.
"""

import json
from itertools import product

# Base shared configuration
base_config = {
    "scale": 0.1,
    "lr": 0.001,
    "mb": 512,
    "epochs": 200,
    "trials": 10,
    "widths": [8192, 8192, 8192],
    "nonlinearity": False,
}

# Generate FastfoodLayer configurations
for learn_S, learn_G, learn_B in product([False, True], repeat=3):
    config = {
        "layer": "FastfoodLayer",
        "hadamard": "Dao",
        "learn_S": learn_S,
        "learn_G": learn_G,
        "learn_B": learn_B
    }
    config.update(base_config)
    filename = f"Fastfood_S{int(learn_S)}_G{int(learn_G)}_B{int(learn_B)}.json"
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)

# Generate RKSLayer configurations
for learn_G in [False, True]:
    config = {
        "layer": "RKSLayer",
        "learn_G": learn_G
    }
    config.update(base_config)
    filename = f"RKS_G{int(learn_G)}.json"
    with open(filename, "w") as f:
        json.dump(config, f, indent=4)

print("All configuration files generated in current directory.")
