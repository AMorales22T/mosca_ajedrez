from __future__ import annotations

import yaml

from .common import PROCESSED, json_dump
from .device import report_device
from .models import CNNControl, FlyBrainNet, MLPControl, load_connectome, parameter_report, shuffle_degree_preserving
from .train import seed_everything, synthetic_dataset, train_model


def run(config_path="configs/smoke.yaml"):
    with open(config_path, encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    seed = int(config["seed"]); seed_everything(seed); report_device()
    data = synthetic_dataset(int(config["n_positions"]), seed)
    split = int(len(data[0]) * 0.8)
    train_data = tuple(value[:split] for value in data); val_data = tuple(value[split:] for value in data)
    bio_mask = load_connectome("synthetic", config["units"], seed)
    models = {
        "fly": FlyBrainNet(config["units"], config["recurrent_steps"], "synthetic", seed, connectome=bio_mask),
        "shuffled": FlyBrainNet(config["units"], config["recurrent_steps"], "synthetic", seed + 1, connectome=shuffle_degree_preserving(bio_mask, seed + 1)),
        "mlp": MLPControl(config["units"], config["recurrent_steps"], seed=seed),
        "cnn": CNNControl(config["units"], config["recurrent_steps"], seed=seed),
    }
    counts = parameter_report(models)
    results = {}
    for name, model in models.items():
        results[name] = train_model(model, train_data, val_data, epochs=config["epochs"], batch_size=config["batch_size"], value_weight=config["value_weight"], run_name=f"smoke_{name}", seed=seed)
    json_dump(PROCESSED / "smoke_report.json", {"connectome_mode": "synthetic", "parameters": counts, "results": results})
    return results


if __name__ == "__main__":
    run()
