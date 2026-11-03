import copy
import logging
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from utils.toolkit import tensor2numpy, accuracy
from scipy.spatial.distance import cdist

EPSILON = 1e-8
batch_size = 128


class BaseLearner(object):
    def __init__(self, args):
        self.args = args
        self._cur_task = -1
        self._known_classes = 0
        self._total_classes = 0
        self._network = None
        self._old_network = None
        self._data_memory, self._targets_memory = np.array([]), np.array([])
        self.topk = 5

        self._memory_size = args["memory_size"]
        self._memory_per_class = args.get("memory_per_class", None)
        self._fixed_memory = args.get("fixed_memory", False)
        self._device = args["device"][0]
        self._multiple_gpus = args["device"]

    @property
    def exemplar_size(self):
        assert len(self._data_memory) == len(
            self._targets_memory
        ), "Exemplar size error."
        return len(self._targets_memory)

    @property
    def samples_per_class(self):
        if self._fixed_memory:
            return self._memory_per_class
        else:
            assert self._total_classes != 0, "Total classes is 0"
            return self._memory_size // self._total_classes

    @property
    def feature_dim(self):
        if isinstance(self._network, nn.DataParallel):
            return self._network.module.feature_dim
        else:
            return self._network.feature_dim

    def build_rehearsal_memory(self, data_manager, per_class):
        if self._fixed_memory:
            self._construct_exemplar_unified(data_manager, per_class)
        else:
            self._reduce_exemplar(data_manager, per_class)
            self._construct_exemplar(data_manager, per_class)

    def save_checkpoint(self, filename):
        self._network.cpu()
        save_dict = {
            "tasks": self._cur_task,
            "model_state_dict": self._network.state_dict(),
        }
        torch.save(save_dict, "{}_{}.pkl".format(filename, self._cur_task))

    def after_task(self):
        pass

    def _evaluate(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._known_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def _evaluate_zs(self, y_pred, y_true):
        ret = {}
        grouped = accuracy(y_pred.T[0], y_true, self._total_classes)
        ret["grouped"] = grouped
        ret["top1"] = grouped["total"]
        ret["top{}".format(self.topk)] = np.around(
            (y_pred.T == np.tile(y_true, (self.topk, 1))).sum() * 100 / len(y_true),
            decimals=2,
        )
        return ret

    def _evaluate_zs_seen_unseen(self, y_pred, y_true):

        top1_pred = y_pred.T[0]
        nb_seen = self._total_classes

        seen_idx = np.where(y_true < nb_seen)[0]

        unseen_idx = np.where(y_true >= nb_seen)[0]

        acc_seen = (
            0.0
            if len(seen_idx) == 0
            else float(
                np.around(
                    (top1_pred[seen_idx] == y_true[seen_idx]).sum()
                    * 100
                    / len(seen_idx),
                    decimals=2,
                )
            )
        )

        has_unseen = len(unseen_idx) > 0
        acc_unseen = (
            0.0
            if not has_unseen
            else float(
                np.around(
                    (top1_pred[unseen_idx] == y_true[unseen_idx]).sum()
                    * 100
                    / len(unseen_idx),
                    decimals=2,
                )
            )
        )

        if not has_unseen or (acc_seen + acc_unseen) == 0:
            harmonic = 0.0
        else:
            harmonic = float(
                np.around(
                    2 * acc_seen * acc_unseen / (acc_seen + acc_unseen), decimals=2
                )
            )

        return {
            "seen": acc_seen,
            "unseen": acc_unseen,
            "harmonic": harmonic,
            "has_unseen": has_unseen,
            "num_seen": int(len(seen_idx)),
            "num_unseen": int(len(unseen_idx)),
        }

    def eval_task(self):
        y_pred, y_true = self._eval_cnn(self.test_loader)
        cnn_accy = self._evaluate(y_pred, y_true)

        if hasattr(self, "_class_means"):
            y_pred, y_true = self._eval_nme(self.test_loader, self._class_means)
            nme_accy = self._evaluate(y_pred, y_true)
        else:
            nme_accy = None

        if (
            self.args["convnet_type"].lower() != "clip"
            or self.args["model_name"].lower() == "l2p"
            or self.args["model_name"].lower() == "dualprompt"
        ):
            return cnn_accy, nme_accy, None, None, None, None
        else:
            y_pred, y_true = self._eval_zero_shot()

            zs_dict = self._evaluate_zs_seen_unseen(y_pred, y_true)
            zs_seen, zs_unseen, zs_harmonic = (
                zs_dict["seen"],
                zs_dict["unseen"],
                zs_dict["harmonic"],
            )
            zs_total = self._evaluate_zs(y_pred, y_true)["grouped"]["total"]

            if zs_dict["has_unseen"]:
                logging.info(
                    "[Zero-Shot] Seen Acc: {:.2f} | Unseen Acc: {:.2f} | Harmonic Mean: {:.2f} "
                    "(#seen={}, #unseen={})".format(
                        zs_seen,
                        zs_unseen,
                        zs_harmonic,
                        zs_dict["num_seen"],
                        zs_dict["num_unseen"],
                    )
                )
            else:
                logging.info(
                    "[Zero-Shot] Last task (no unseen classes) => only Seen Acc: {:.2f} "
                    "(Unseen/HM skipped) (#seen={})".format(
                        zs_seen, zs_dict["num_seen"]
                    )
                )

        return cnn_accy, nme_accy, zs_seen, zs_unseen, zs_harmonic, zs_total

    def _eval_zero_shot(self):
        self._network.eval()
        class_to_label = self.data_manager._class_to_label
        templates = self.data_manager._data_to_prompt
        total_labels = class_to_label
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

        test_dataset = self.data_manager.get_dataset(
            np.arange(0, len(total_labels)), source="test", mode="test"
        )
        loader = DataLoader(
            test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.args.get("num_workers", 8),
        )

        y_pred, y_true = [], []
        logits = []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                image_features = self._network.encode_image(inputs)
                image_features /= image_features.norm(dim=-1, keepdim=True)
                outputs = image_features @ text_features.T
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())
            logits.append(outputs.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def incremental_train(self):
        pass

    def _train(self):
        pass

    def _get_memory(self):
        if len(self._data_memory) == 0:
            return None
        else:
            return (self._data_memory, self._targets_memory)

    def _compute_accuracy(self, model, loader):
        model.eval()
        correct, total = 0, 0
        for i, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = model(inputs)["logits"]
            predicts = torch.max(outputs, dim=1)[1]
            correct += (predicts.cpu() == targets).sum()
            total += len(targets)

        return np.around(tensor2numpy(correct) * 100 / total, decimals=2)

    def _eval_cnn(self, loader):
        self._network.eval()
        y_pred, y_true = [], []
        for _, (_, inputs, targets) in enumerate(loader):
            inputs = inputs.to(self._device)
            with torch.no_grad():
                outputs = self._network(inputs)["logits"]
            predicts = torch.topk(
                outputs, k=self.topk, dim=1, largest=True, sorted=True
            )[1]
            y_pred.append(predicts.cpu().numpy())
            y_true.append(targets.cpu().numpy())

        return np.concatenate(y_pred), np.concatenate(y_true)

    def _eval_nme(self, loader, class_means):
        self._network.eval()
        vectors, y_true = self._extract_vectors(loader)
        vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T

        dists = cdist(class_means, vectors, "sqeuclidean")
        scores = dists.T

        return np.argsort(scores, axis=1)[:, : self.topk], y_true

    def _extract_vectors(self, loader):
        self._network.eval()
        vectors, targets = [], []
        with torch.no_grad():
            for _, _inputs, _targets in loader:
                _targets = _targets.numpy()
                if isinstance(self._network, nn.DataParallel):
                    _vectors = tensor2numpy(
                        self._network.module.extract_vector(_inputs.to(self._device))
                    )
                else:
                    _vectors = tensor2numpy(
                        self._network.extract_vector(_inputs.to(self._device))
                    )
                vectors.append(_vectors)
                targets.append(_targets)

        return np.concatenate(vectors), np.concatenate(targets)

    def _reduce_exemplar(self, data_manager, m):
        logging.info("Reducing exemplars...({} per classes)".format(m))
        dummy_data, dummy_targets = copy.deepcopy(self._data_memory), copy.deepcopy(
            self._targets_memory
        )
        self._class_means = np.zeros((self._total_classes, self.feature_dim))
        self._data_memory, self._targets_memory = np.array([]), np.array([])

        for class_idx in range(self._known_classes):
            mask = np.where(dummy_targets == class_idx)[0]
            dd, dt = dummy_data[mask][:m], dummy_targets[mask][:m]
            self._data_memory = (
                np.concatenate((self._data_memory, dd))
                if len(self._data_memory) != 0
                else dd
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, dt))
                if len(self._targets_memory) != 0
                else dt
            )

            idx_dataset = data_manager.get_dataset(
                [], source="train", mode="test", appendent=(dd, dt)
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.get("num_workers", 8),
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar(self, data_manager, m):
        logging.info("Constructing exemplars...({} per classes)".format(m))
        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, idx_dataset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.get("num_workers", 8),
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            m = min(m, len(vectors))
            if m == 0:
                logging.warning(
                    "Class %d has no training samples, skipping exemplar construction.",
                    class_idx,
                )
                continue

            remaining_original_indices = np.arange(len(data), dtype=np.int64)

            selected_exemplars = []
            exemplar_vectors = []
            selected_original_indices = []
            selected_mean_gaps = []

            for k in range(1, m + 1):
                if len(exemplar_vectors) == 0:
                    selected_sum = np.zeros(vectors.shape[1], dtype=vectors.dtype)
                else:
                    selected_sum = np.sum(exemplar_vectors, axis=0)

                candidate_means = (vectors + selected_sum) / k
                candidate_gaps = np.sqrt(
                    np.sum((class_mean - candidate_means) ** 2, axis=1)
                )
                local_idx = int(np.argmin(candidate_gaps))
                original_idx = int(remaining_original_indices[local_idx])

                selected_exemplars.append(np.array(data[local_idx]))
                exemplar_vectors.append(np.array(vectors[local_idx]))
                selected_original_indices.append(original_idx)
                selected_mean_gaps.append(float(candidate_gaps[local_idx]))

                vectors = np.delete(vectors, local_idx, axis=0)
                data = np.delete(data, local_idx, axis=0)
                remaining_original_indices = np.delete(
                    remaining_original_indices,
                    local_idx,
                    axis=0,
                )

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            idx_dataset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            idx_loader = DataLoader(
                idx_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.get("num_workers", 8),
            )
            vectors, _ = self._extract_vectors(idx_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)

            self._class_means[class_idx, :] = mean

    def _construct_exemplar_unified(self, data_manager, m):
        logging.info(
            "Constructing exemplars for new classes...({} per classes)".format(m)
        )
        _class_means = np.zeros((self._total_classes, self.feature_dim))

        for class_idx in range(self._known_classes):
            mask = np.where(self._targets_memory == class_idx)[0]
            class_data, class_targets = (
                self._data_memory[mask],
                self._targets_memory[mask],
            )

            class_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(class_data, class_targets),
            )
            class_loader = DataLoader(
                class_dset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.get("num_workers", 8),
            )
            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean

        for class_idx in range(self._known_classes, self._total_classes):
            data, targets, class_dset = data_manager.get_dataset(
                np.arange(class_idx, class_idx + 1),
                source="train",
                mode="test",
                ret_data=True,
            )
            class_loader = DataLoader(
                class_dset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.get("num_workers", 8),
            )

            vectors, _ = self._extract_vectors(class_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            class_mean = np.mean(vectors, axis=0)

            m_actual = min(m, len(vectors))
            if m_actual == 0:
                logging.warning(
                    "Class %d has no training samples, skipping exemplar construction.",
                    class_idx,
                )
                continue

            remaining_original_indices = np.arange(len(data), dtype=np.int64)

            selected_exemplars = []
            exemplar_vectors = []
            selected_original_indices = []
            selected_mean_gaps = []

            for k in range(1, m_actual + 1):
                if len(exemplar_vectors) == 0:
                    selected_sum = np.zeros(vectors.shape[1], dtype=vectors.dtype)
                else:
                    selected_sum = np.sum(exemplar_vectors, axis=0)

                candidate_means = (vectors + selected_sum) / k
                candidate_gaps = np.sqrt(
                    np.sum((class_mean - candidate_means) ** 2, axis=1)
                )
                local_idx = int(np.argmin(candidate_gaps))
                original_idx = int(remaining_original_indices[local_idx])

                selected_exemplars.append(np.array(data[local_idx]))
                exemplar_vectors.append(np.array(vectors[local_idx]))
                selected_original_indices.append(original_idx)
                selected_mean_gaps.append(float(candidate_gaps[local_idx]))

                vectors = np.delete(vectors, local_idx, axis=0)
                data = np.delete(data, local_idx, axis=0)
                remaining_original_indices = np.delete(
                    remaining_original_indices,
                    local_idx,
                    axis=0,
                )

            selected_exemplars = np.array(selected_exemplars)
            exemplar_targets = np.full(m_actual, class_idx)
            self._data_memory = (
                np.concatenate((self._data_memory, selected_exemplars))
                if len(self._data_memory) != 0
                else selected_exemplars
            )
            self._targets_memory = (
                np.concatenate((self._targets_memory, exemplar_targets))
                if len(self._targets_memory) != 0
                else exemplar_targets
            )

            exemplar_dset = data_manager.get_dataset(
                [],
                source="train",
                mode="test",
                appendent=(selected_exemplars, exemplar_targets),
            )
            exemplar_loader = DataLoader(
                exemplar_dset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=self.args.get("num_workers", 8),
            )
            vectors, _ = self._extract_vectors(exemplar_loader)
            vectors = (vectors.T / (np.linalg.norm(vectors.T, axis=0) + EPSILON)).T
            mean = np.mean(vectors, axis=0)
            mean = mean / np.linalg.norm(mean)
            _class_means[class_idx, :] = mean

        self._class_means = _class_means
