import argparse
import json
from pathlib import Path


def load_config(path):
    with Path(path).open(encoding="utf-8") as handle:
        config = json.load(handle)
    if config.get("model_name", "").lower() != "gkgla":
        raise ValueError("model_name must be gkgla")
    for key in (
        "init_cls",
        "increment",
        "batch_size",
        "tuned_epoch",
        "lora_rank",
        "fusion_num_layers",
    ):
        if int(config[key]) < 1:
            raise ValueError(f"{key} must be positive")
    if not isinstance(config.get("seed"), list) or not config["seed"]:
        raise ValueError("seed must be a nonempty list of class-order seeds")
    return config


def main():
    parser = argparse.ArgumentParser(description="GKGLA class-incremental learning")
    parser.add_argument(
        "--config", default=str(Path(__file__).parent / "configs/cifar.json")
    )
    parser.add_argument("--data-root")
    parser.add_argument("--clip-checkpoint")
    parser.add_argument("--device", help="CUDA index, cuda:0, cpu, or -1")
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--check-config", action="store_true")
    cli = parser.parse_args()
    config = load_config(cli.config)
    for key in ("data_root", "clip_checkpoint", "num_workers"):
        value = getattr(cli, key)
        if value is not None:
            config[key] = value
    if cli.device is not None:
        config["device"] = [cli.device]
    if cli.check_config:
        print(json.dumps(config, indent=2))
        return
    from trainer import train

    train(config)


if __name__ == "__main__":
    main()
