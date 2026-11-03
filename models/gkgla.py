import logging
import numpy as np
import torch
from torch import nn
from tqdm import tqdm
from torch import optim
from torch.nn import functional as F
from torch.utils.data import DataLoader
from models.network import GKGLANet
from models.base import BaseLearner
from utils.toolkit import tensor2numpy, get_attribute, ClipLoss
import os

num_workers = 8


class GKGLALearner(BaseLearner):
    def __init__(self, args):
        super().__init__(args)
        self.args = args

        self._train_transformer = False

        self._network = GKGLANet(args, False)

        self.batch_size = get_attribute(args, "batch_size", 64)
        self.init_lr = get_attribute(args, "init_lr", 0.01)
        self.weight_decay = get_attribute(args, "weight_decay", 0.0005)
        self.min_lr = get_attribute(args, "min_lr", 1e-8)
        self.frozen_layers = get_attribute(args, "frozen_layers", None)

        self.tuned_epoch = get_attribute(args, "tuned_epoch", 5)

        self._known_classes = 0
        self.use_cos = get_attribute(args, "use_cos", False)

    def _unwrap_network(self):
        return (
            self._network.module
            if isinstance(self._network, nn.DataParallel)
            else self._network
        )

    def _group_name_by_param_name(self, name: str) -> str:

        if name.startswith("module."):
            name = name[len("module.") :]

        if name.startswith("convnet."):
            return "convnet"

        if name.startswith("base_proj_img."):
            return "base_proj_img"
        if name.startswith("base_proj_text."):
            return "base_proj_text"

        if name.startswith("projs_img."):
            return "projs_img"
        if name.startswith("projs_text."):
            return "projs_text"

        if (
            name.startswith("sel_attn_layers.")
            or name.startswith("fuse_norm_layers.")
            or name.startswith("gamma_layers.")
            or name.startswith("pseudo_queries.")
            or name.startswith("attn_res_norms.")
        ):
            return "fusion"
        if name.startswith("context_prompts."):
            return "context_prompts"
        return "others"

    def _collect_param_stats_by_name(self, trainable_only: bool = False):
        net = self._unwrap_network()
        keys = [
            "convnet",
            "base_proj_img",
            "base_proj_text",
            "projs_img",
            "projs_text",
            "fusion",
            "context_prompts",
            "others",
        ]

        stats = {k: {"numel": 0, "tensors": 0} for k in keys}

        for n, p in net.named_parameters():
            if trainable_only and (not p.requires_grad):
                continue
            g = self._group_name_by_param_name(n)
            stats[g]["numel"] += p.numel()
            stats[g]["tensors"] += 1

        total_numel = sum(v["numel"] for v in stats.values())
        total_tensors = sum(v["tensors"] for v in stats.values())
        return stats, total_numel, total_tensors

    def _log_param_stats_by_name(self, stage_tag: str = ""):
        all_stats, all_numel, all_tensors = self._collect_param_stats_by_name(
            trainable_only=False
        )
        tr_stats, tr_numel, tr_tensors = self._collect_param_stats_by_name(
            trainable_only=True
        )

        order = [
            "convnet",
            "base_proj_img",
            "base_proj_text",
            "projs_img",
            "projs_text",
            "fusion",
            "context_prompts",
            "others",
        ]

        def fmt(d):
            return ", ".join(
                [f"{k}: {d[k]['numel']}({d[k]['tensors']}t)" for k in order]
            )

        prefix = f"[{stage_tag}] " if stage_tag else ""
        logging.info(
            "%sParams(all) => %s, total: %d(%dt)",
            prefix,
            fmt(all_stats),
            all_numel,
            all_tensors,
        )
        logging.info(
            "%sParams(trainable) => %s, total: %d(%dt)",
            prefix,
            fmt(tr_stats),
            tr_numel,
            tr_tensors,
        )

    def after_task(self):
        self._known_classes = self._total_classes
        logging.info("Exemplar size: {}".format(self.exemplar_size))

    def _save_stage_checkpoint(self):
        if not bool(self.args.get("save_stage_checkpoints", False)):
            return
        save_dir = getattr(self, "save_dir", None)
        if save_dir is None:
            save_dir = self.args.get("checkpoint_dir", "checkpoints")
        ckpt_dir = os.path.join(save_dir, "checkpoints")
        os.makedirs(ckpt_dir, exist_ok=True)
        net = self._unwrap_network()
        ckpt_path = os.path.join(
            ckpt_dir,
            "task_{}_classes_{}.pth".format(self._cur_task, self._total_classes),
        )
        torch.save(
            {
                "task": self._cur_task,
                "total_classes": self._total_classes,
                "known_classes": self._known_classes,
                "model_state_dict": net.state_dict(),
                "args": self.args,
            },
            ckpt_path,
        )
        logging.info("[Checkpoint] Saved stage checkpoint: %s", ckpt_path)

    def cal_prototype(self, trainloader, model):

        model = model.eval()

        embedding_list = []
        label_list = []

        with torch.no_grad():
            for i, batch in enumerate(trainloader):
                _, data, label = batch
                data = data.to(self._device)
                label = label.to(self._device)
                embedding = model.convnet.encode_image(data, True)
                embedding_list.append(embedding.cpu())
                label_list.append(label.cpu())
        embedding_list = torch.cat(embedding_list, dim=0)
        label_list = torch.cat(label_list, dim=0)

        class_list = list(range(self._known_classes, self._total_classes))
        for class_index in class_list:
            data_index = (label_list == class_index).nonzero().squeeze(-1)
            embedding = embedding_list[data_index]
            proto = embedding.mean(0)
            self._network.img_prototypes[class_index] = proto

    def incremental_train(self, data_manager):
        self._cur_task += 1
        self._total_classes = self._known_classes + data_manager.get_task_size(
            self._cur_task
        )

        self._network.update_prototype(self._total_classes)
        self._network.update_context_prompt()

        self._network.extend_task()

        logging.info(
            "Learning on {}-{}".format(self._known_classes, self._total_classes)
        )

        train_dataset = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="train",
            appendent=self._get_memory(),
        )
        self.train_dataset = train_dataset
        self.data_manager = data_manager
        self._network.to(self._device)

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.get("num_workers", 8),
        )

        test_dataset = data_manager.get_dataset(
            np.arange(0, self._total_classes), source="test", mode="test"
        )
        self.test_loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.get("num_workers", 8),
        )

        train_dataset_for_protonet = data_manager.get_dataset(
            np.arange(self._known_classes, self._total_classes),
            source="train",
            mode="test",
        )
        self.train_loader_for_protonet = DataLoader(
            train_dataset_for_protonet,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.args.get("num_workers", 8),
        )

        if len(self._multiple_gpus) > 1:
            print("Multiple GPUs")
            self._network = nn.DataParallel(self._network, self._multiple_gpus)

        self.cal_prototype(self.train_loader_for_protonet, self._network)
        self._train_proj(
            self.train_loader, self.test_loader, self.train_loader_for_protonet
        )
        self.build_rehearsal_memory(data_manager, self.samples_per_class)
        self._save_stage_checkpoint()
        if len(self._multiple_gpus) > 1:
            self._network = self._network.module

    def _train_proj(self, train_loader, test_loader, train_loader_for_protonet):
        self._train_transformer = True
        self._network.to(self._device)
        net = self._unwrap_network()

        frozen_cnt = 0
        trainable_cnt = 0

        for name, param in net.convnet.named_parameters():
            if "logit_scale" not in name:
                param.requires_grad = False
                frozen_cnt += 1
            else:
                param.requires_grad = True
                trainable_cnt += 1

        net.freeze_projection_weight_new()

        logging.info(
            "Freeze summary => convnet tensors frozen/trainable: %d/%d",
            frozen_cnt,
            trainable_cnt,
        )
        self._log_param_stats_by_name(stage_tag=f"task{self._cur_task}_after_freeze")

        logging.info("===== Trainable parameter names after freeze =====")
        for n, p in net.named_parameters():
            if p.requires_grad:
                logging.info("%s | shape=%s | numel=%d", n, tuple(p.shape), p.numel())

        named_trainable = [(n, p) for n, p in net.named_parameters() if p.requires_grad]

        g_base, g_lora, g_prompt, g_norm_gamma, g_attn, g_query, g_other = (
            [],
            [],
            [],
            [],
            [],
            [],
            [],
        )

        for n, p in named_trainable:
            name = n.lower()

            if name.startswith("base_proj_img.") or name.startswith("base_proj_text."):
                g_base.append(p)
            elif name.startswith("projs_img.") or name.startswith("projs_text."):
                g_lora.append(p)
            elif name.startswith("context_prompts."):
                g_prompt.append(p)
            elif "norm" in name or "gamma" in name:
                g_norm_gamma.append(p)
            elif "query" in name or "pseudo" in name:
                g_query.append(p)
            elif "attn" in name:
                g_attn.append(p)
            else:
                g_other.append(p)

        lr_lora = self.init_lr * 5.0
        lr_attn = self.init_lr * 0.5
        lr_norm_gamma = self.init_lr * 0.1
        lr_prompt = self.init_lr * 0.2
        lr_query = self.init_lr * 1.0

        param_groups = []

        if len(g_base) > 0:
            param_groups.append(
                {"params": g_base, "lr": self.init_lr, "weight_decay": 0.0}
            )

        if len(g_lora) > 0:
            param_groups.append(
                {"params": g_lora, "lr": lr_lora, "weight_decay": self.weight_decay}
            )

        if len(g_attn) > 0:
            param_groups.append(
                {"params": g_attn, "lr": lr_attn, "weight_decay": self.weight_decay}
            )

        if len(g_norm_gamma) > 0:
            param_groups.append(
                {"params": g_norm_gamma, "lr": lr_norm_gamma, "weight_decay": 0.0}
            )

        if len(g_prompt) > 0:
            param_groups.append(
                {"params": g_prompt, "lr": lr_prompt, "weight_decay": 0.0}
            )

        if len(g_other) > 0:
            param_groups.append(
                {
                    "params": g_other,
                    "lr": self.init_lr * 0.3,
                    "weight_decay": self.weight_decay,
                }
            )

        if len(g_query) > 0:
            param_groups.append(
                {"params": g_query, "lr": lr_query, "weight_decay": 0.0}
            )

        optimizer = optim.SGD(param_groups, momentum=0.9, nesterov=True)

        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.args["tuned_epoch"], eta_min=1e-6
        )

        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt[0]
        prog_bar = tqdm(range(self.tuned_epoch))
        cliploss = ClipLoss()

        total_labels = class_to_label[: self._total_classes]
        for _, epoch in enumerate(prog_bar):
            self._network.train()
            losses = 0.0
            correct, total = 0, 0
            for i, (_, inputs, targets) in enumerate(train_loader):
                labels = [class_to_label[y] for y in targets]
                inputs = inputs.to(self._device)
                targets = targets.to(self._device)

                texts = [templates.format(inst) for inst in total_labels]
                texts = self._network.tokenizer(texts).to(self._device)
                text_features = self._network.encode_text(texts)
                text_feas = text_features / text_features.norm(dim=-1, keepdim=True)

                image_features = self._network.encode_image(inputs)
                img_feas = image_features / image_features.norm(dim=-1, keepdim=True)

                image_features, text_features, logit_scale, proto_feas = (
                    self._network.forward_transformer(
                        img_feas, text_feas, self._train_transformer
                    )
                )

                logits = image_features @ text_features.T

                texts = [templates.format(inst) for inst in labels]
                clip_text_feas = self._network.encode_text(
                    self._network.tokenizer(texts).to(self._device)
                )
                clip_text_norm = clip_text_feas.norm(dim=-1, keepdim=True)
                clip_text_feas = clip_text_feas / clip_text_norm

                clip_loss = cliploss(img_feas, clip_text_feas, logit_scale)

                loss = F.cross_entropy(logits, targets)

                protoloss = F.cross_entropy(image_features @ proto_feas.T, targets)

                total_loss = loss + clip_loss + protoloss

                optimizer.zero_grad()
                total_loss.backward()
                optimizer.step()
                losses += total_loss.item()

                _, preds = torch.max(logits, dim=1)
                correct += preds.eq(targets.expand_as(preds)).cpu().sum()
                total += len(targets)

            scheduler.step()
            train_acc = np.around(tensor2numpy(correct) * 100 / total, decimals=2)
            test_acc = self._compute_accuracy(self._network, test_loader)
            info = "Task {}, Epoch {}/{} => Loss {:.3f}, Train_acc {:.2f}, Test_acc {:.2f}".format(
                self._cur_task,
                epoch + 1,
                self.args["tuned_epoch"],
                losses / len(train_loader),
                train_acc,
                test_acc,
            )
            prog_bar.set_description(info)

    def _compute_accuracy(self, model, loader):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[: self._total_classes]
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():

                image_features = self._network.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = (
                    self._network.forward_transformer(
                        image_features, text_features, self._train_transformer
                    )
                )
                outputs = transf_image_features @ transf_text_features.T
                proto_outputs = transf_image_features @ proto_feas.T
                original_outputs = image_features @ text_features.T
                outputs = original_outputs + outputs + proto_outputs

            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)
        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):

        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label[: self._total_classes]
        text_features = []
        with torch.no_grad():
            for l in total_labels:
                texts = [t.format(l) for t in templates]
                texts = self._network.tokenizer(texts).to(self._device)
                class_embeddings = self._network.encode_text(texts)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                class_embeddings = class_embeddings.mean(dim=0)
                class_embeddings = class_embeddings / class_embeddings.norm(
                    dim=-1, keepdim=True
                )
                text_features.append(class_embeddings)
            text_features = torch.stack(text_features, dim=0)

        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features = self._network.encode_image(inputs)
                transf_image_features, transf_text_features, _, proto_feas = (
                    self._network.forward_transformer(
                        image_features, text_features, self._train_transformer
                    )
                )
                outputs = transf_image_features @ transf_text_features.T

                proto_outputs = transf_image_features @ proto_feas.T

                original_outputs = image_features @ text_features.T

                outputs = original_outputs + outputs + proto_outputs

            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)
