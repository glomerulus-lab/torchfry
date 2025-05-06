import json
import os
from itertools import product

# Base shared config
base_config = {
    "scale": 0.1,
    "lr": 0.1,
    "mb": 1024,
    "epochs": 200,
    "trials": 5,
    "features": 512,
    "nonlinearity": False
}

# Generate FastFoodLayer combinations
for learn_S, learn_G, learn_B in product([False, True], repeat=3):
    config = {
        "layer": "FastFoodLayer",
        "hadamard": "Dao",
        "learn_S": learn_S,
        "learn_G": learn_G,
        "learn_B": learn_B
    }
    config.update(base_config)
    name = f"FastFood_S{int(learn_S)}_G{int(learn_G)}_B{int(learn_B)}.json"
    with open(name, "w") as f:
        json.dump(config, f, indent=4)

# Generate RKSLayer combinations
for learn_G in [False, True]:
    config = {
        "layer": "RKSLayer",
        "learn_G": learn_G
    }
    config.update(base_config)
    name = f"RKS_G{int(learn_G)}.json"
    with open(name, "w") as f:
        json.dump(config, f, indent=4)

print("All configuration files generated in current directory.")
