# ------------------------------------------------------------------------------
# Custom dataset adapter for the robotics semantic masks exported from Roboflow.
# ------------------------------------------------------------------------------
import os

import numpy as np
from PIL import Image
import torch

from .base_dataset import BaseDataset


class RoboticsSemantic(BaseDataset):
    def __init__(
        self,
        root,
        list_path,
        num_classes=6,
        multi_scale=True,
        flip=True,
        ignore_label=255,
        base_size=432,
        crop_size=(432, 432),
        scale_factor=16,
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
        bd_dilate_size=4,
    ):
        super(RoboticsSemantic, self).__init__(
            ignore_label, base_size, crop_size, scale_factor, mean, std
        )
        self.root = root
        self.list_path = list_path
        self.num_classes = num_classes
        self.multi_scale = multi_scale
        self.flip = flip
        self.ignore_label = ignore_label
        self.bd_dilate_size = bd_dilate_size
        self.class_weights = None
        self.class_names = [
            "background",
            "human",
            "obstacle",
            "road",
            "sidewalk",
            "speed_breaker",
        ]
        self.color_list = [
            [20, 24, 33],
            [51, 122, 183],
            [64, 145, 108],
            [229, 126, 49],
            [214, 48, 49],
            [243, 196, 15],
        ]
        self.label_mapping = {
            0: 0,
            1: 1,
            2: 2,
            3: 3,
            4: 4,
            5: 5,
            6: ignore_label,
            7: ignore_label,
            255: ignore_label,
        }

        list_file = list_path if os.path.isabs(list_path) else os.path.join(root, list_path)
        self.img_list = [line.strip().split() for line in open(list_file) if line.strip()]
        self.files = self.read_files()
        self.class_weights = self._compute_class_weights()

    def _resolve_path(self, path):
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def read_files(self):
        files = []
        for item in self.img_list:
            image_path, label_path = item
            name = os.path.splitext(os.path.basename(label_path))[0]
            files.append({"img": image_path, "label": label_path, "name": name})
        return files

    def convert_label(self, label):
        converted = np.ones_like(label, dtype=np.uint8) * self.ignore_label
        for raw_label, train_label in self.label_mapping.items():
            converted[label == raw_label] = train_label
        return converted

    def label2color(self, label):
        color_map = np.zeros(label.shape + (3,), dtype=np.uint8)
        for label_id, color in enumerate(self.color_list):
            color_map[label == label_id] = color
        return color_map

    def _compute_class_weights(self):
        pixel_counts = np.zeros(self.num_classes, dtype=np.float64)
        for item in self.files:
            label = Image.open(self._resolve_path(item["label"])).convert("L")
            converted = self.convert_label(np.array(label))
            valid = converted != self.ignore_label
            if not np.any(valid):
                continue
            counts = np.bincount(converted[valid].ravel(), minlength=self.num_classes)
            pixel_counts += counts[: self.num_classes]

        if pixel_counts.sum() <= 0:
            return None

        class_prob = pixel_counts / pixel_counts.sum()
        weights = 1.0 / np.log(1.02 + class_prob)
        weights = weights / weights.mean()
        return torch.tensor(weights, dtype=torch.float32)

    def __getitem__(self, index):
        item = self.files[index]
        name = item["name"]
        image = Image.open(self._resolve_path(item["img"])).convert("RGB")
        image = np.array(image)
        size = image.shape
        label = Image.open(self._resolve_path(item["label"])).convert("L")
        label = self.convert_label(np.array(label))
        image, label, edge = self.gen_sample(
            image,
            label,
            self.multi_scale,
            self.flip,
            edge_pad=False,
            edge_size=self.bd_dilate_size,
            city=False,
        )
        return image.copy(), label.copy(), edge.copy(), np.array(size), name

    def single_scale_inference(self, config, model, image):
        return self.inference(config, model, image)

    def save_pred(self, preds, sv_path, name):
        preds = np.asarray(np.argmax(preds.cpu(), axis=1), dtype=np.uint8)
        for index in range(preds.shape[0]):
            pred = self.label2color(preds[index])
            save_img = Image.fromarray(pred)
            save_img.save(os.path.join(sv_path, name[index] + ".png"))
