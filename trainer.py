import sys
import logging
import copy
import torch
from utils import factory
from utils.data_manager import DataManager
from utils.toolkit import count_parameters
import os


def train(args):
    seed_list = copy.deepcopy(args["seed"])
    device = copy.deepcopy(args["device"])

    for seed in seed_list:
        args["seed"] = seed
        args["device"] = device
        _train(args)


def _train(args):

    init_cls = 0 if args["init_cls"] == args["increment"] else args["init_cls"]

    logs_name = "logs/{}/{}/{}/{}".format(
        args["model_name"], args["dataset"], init_cls, args["increment"]
    )

    if not os.path.exists(logs_name):
        os.makedirs(logs_name)

    logfilename = "logs/{}/{}/{}/{}/{}_{}_{}".format(
        args["model_name"],
        args["dataset"],
        init_cls,
        args["increment"],
        args["prefix"],
        args["seed"],
        args["convnet_type"],
    )

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(filename)s] => %(message)s",
        handlers=[
            logging.FileHandler(filename=logfilename + ".log"),
            logging.StreamHandler(sys.stdout),
        ],
        force=True,
    )

    _set_random(int(args.get("training_seed", 1)))
    _set_device(args)
    print_args(args)

    data_manager = DataManager(
        args["dataset"],
        args["shuffle"],
        args["seed"],
        args["init_cls"],
        args["increment"],
        data_root=args.get("data_root", "data"),
    )

    model = factory.get_model(args["model_name"], args)
    model.save_dir = logs_name

    cnn_curve = {"top1": [], "top5": []}
    zs_seen_curve, zs_unseen_curve, zs_harmonic_curve, zs_total_curve = (
        {"top1": [], "top5": []},
        {"top1": [], "top5": []},
        {"top1": [], "top5": []},
        {"top1": [], "top5": []},
    )

    for task in range(data_manager.nb_tasks):
        logging.info("All params: {}".format(count_parameters(model._network)))
        logging.info(
            "Trainable params: {}".format(count_parameters(model._network, True))
        )

        model.incremental_train(data_manager)

        cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total = (
            model.eval_task()
        )

        model.after_task()

        logging.info("CNN: {}".format(cnn_accy["grouped"]))

        cnn_curve["top1"].append(cnn_accy["top1"])
        cnn_curve["top5"].append(cnn_accy["top5"])

        logging.info("CNN top1 curve: {}".format(cnn_curve["top1"]))
        logging.info("CNN top5 curve: {}\n".format(cnn_curve["top5"]))

        print(
            "Average Accuracy (CNN):", sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
        )
        logging.info(
            "Average Accuracy (CNN): {}".format(
                sum(cnn_curve["top1"]) / len(cnn_curve["top1"])
            )
        )

        if zs_seen is not None:
            zs_seen_curve["top1"].append(zs_seen)
            zs_total_curve["top1"].append(zs_total)

            has_unseen = (zs_unseen is not None) and (zs_unseen > 0 or zs_harmonic > 0)
            if has_unseen:
                zs_unseen_curve["top1"].append(zs_unseen)
                zs_harmonic_curve["top1"].append(zs_harmonic)
                logging.info(
                    "Zero-Shot => Seen: {} | Unseen: {} | Harmonic Mean: {} | Total: {}".format(
                        zs_seen, zs_unseen, zs_harmonic, zs_total
                    )
                )
            else:
                logging.info(
                    "Zero-Shot (last task, no unseen classes) => Seen: {} | Total: {}".format(
                        zs_seen, zs_total
                    )
                )
            logging.info("Zero-Shot Seen curve: {}".format(zs_seen_curve["top1"]))
            logging.info("Zero-Shot Unseen curve: {}".format(zs_unseen_curve["top1"]))
            logging.info(
                "Zero-Shot Harmonic Mean curve: {}\n".format(zs_harmonic_curve["top1"])
            )


def _set_device(args):
    devices = args["device"]
    if not isinstance(devices, list):
        devices = [devices]
    if len(devices) != 1:
        raise ValueError("This release supports one device per experiment")
    value = str(devices[0])
    if value in {"-1", "cpu"}:
        args["device"] = [torch.device("cpu")]
    else:
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA unavailable; select device [-1] for CPU")
        args["device"] = [
            torch.device(value if value.startswith("cuda:") else f"cuda:{value}")
        ]


def _set_random(seed=1):
    import random
    import numpy as np

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def print_args(args):

    for key, value in args.items():
        logging.info("{}: {}".format(key, value))
