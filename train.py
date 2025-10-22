#!/usr/bin/env python3
import argparse
import json
import math
import os
import random
import time
from collections import defaultdict

import torch
import torch.nn.functional as F
from torch import nn, optim
from torch.utils.data import Dataset, DataLoader
from torchvision import datasets, transforms, models

from sceneGraphEncodingNet.nets import CSMG, JointNet
from sceneGraphEncodingNet import model_eval as eval_utils


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def build_backbone(name: str, freeze: bool):
    name = name.lower()
    if name == "vgg16":
        weights = models.VGG16_Weights.IMAGENET1K_V1
        backbone = models.vgg16(weights=weights)
        features = list(backbone.features.children())[:-8]  # up to conv4
        backbone = nn.Sequential(*features)
        out_channels = 512
    elif name == "resnet50":
        weights = models.ResNet50_Weights.IMAGENET1K_V1
        backbone = models.resnet50(weights=weights)
        blocks = [
            backbone.conv1,
            backbone.bn1,
            backbone.relu,
            backbone.maxpool,
            backbone.layer1,
            backbone.layer2,
            backbone.layer3,
        ]
        backbone = nn.Sequential(*blocks)
        out_channels = 1024
    else:
        raise ValueError(f"Unsupported backbone {name}")

    if freeze:
        for p in backbone.parameters():
            p.requires_grad = False
    return backbone, out_channels


def unwrap(model):
    return model.module if isinstance(model, nn.DataParallel) else model


def forward_descriptors(model, images):
    net = unwrap(model)
    features = net.backbone(images)
    csmg: CSMG = net.module

    x_nl = csmg.nl_conv(features)
    b, d, h, w = x_nl.shape
    global_desc = F.normalize(x_nl.reshape(b, -1), p=2, dim=1)

    x = F.normalize(x_nl, p=2, dim=1)
    soft_assign = csmg.conv_node(x).view(b, csmg.num_clusters, -1)
    soft_assign = F.softmax(soft_assign, dim=1)

    x_flat = x.view(b, d, -1)
    centroids = F.normalize(csmg.centroids, p=2, dim=1)
    centroids = centroids.unsqueeze(0).expand(b, -1, -1)
    sim = torch.bmm(centroids, x_flat)
    sim = csmg.relu(sim)

    sim_mask = sim.unsqueeze(2).expand(-1, -1, d, -1)
    x_expand = x_flat.unsqueeze(1).expand(-1, csmg.num_clusters, -1, -1)
    x_star = x_expand * soft_assign.unsqueeze(2) * sim_mask

    d_tensor = x_star.sum(dim=-1)
    d_tensor = F.normalize(d_tensor, p=2, dim=2)
    semi_desc = F.normalize(d_tensor.view(b, -1), p=2, dim=1)
    return global_desc, semi_desc


def cosine_triplet_loss(anchor, positive, negative, margin):
    pos = F.cosine_similarity(anchor, positive)
    neg = F.cosine_similarity(anchor, negative)
    loss = F.relu(margin + neg - pos)
    return loss.mean()


class TripletDataset(Dataset):
    def __init__(self, sat_root, drone_root, transform, length):
        self.sat_dataset = datasets.ImageFolder(sat_root, transform=transform)
        self.drone_dataset = datasets.ImageFolder(drone_root, transform=transform)
        if self.sat_dataset.classes != self.drone_dataset.classes:
            raise RuntimeError("Class names between satellite and drone folders do not match.")

        self.classes = list(range(len(self.sat_dataset.classes)))
        self.length = length

        self.sat_by_class = defaultdict(list)
        for idx, (_, cls) in enumerate(self.sat_dataset.samples):
            self.sat_by_class[cls].append(idx)

        self.drone_by_class = defaultdict(list)
        for idx, (_, cls) in enumerate(self.drone_dataset.samples):
            self.drone_by_class[cls].append(idx)

        for cls in self.classes:
            if not self.sat_by_class[cls] or not self.drone_by_class[cls]:
                raise RuntimeError(f"Class {cls} missing satellite or drone samples.")

    def __len__(self):
        return self.length

    def __getitem__(self, _):
        anchor_cls = random.choice(self.classes)
        negative_cls = random.choice([c for c in self.classes if c != anchor_cls])

        anchor_img, _ = self.sat_dataset[random.choice(self.sat_by_class[anchor_cls])]
        positive_img, _ = self.drone_dataset[random.choice(self.drone_by_class[anchor_cls])]
        negative_img, _ = self.drone_dataset[random.choice(self.drone_by_class[negative_cls])]
        return anchor_img, positive_img, negative_img, anchor_cls


def build_transforms(image_size=224):
    train_tf = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop((image_size, image_size), padding=16, padding_mode="reflect"),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.2, 0.2, 0.2, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    test_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    return train_tf, test_tf


def save_checkpoint(state, checkpoint_dir, prefix):
    os.makedirs(checkpoint_dir, exist_ok=True)
    path = os.path.join(checkpoint_dir, f"{prefix}.pth")
    torch.save(state, path)
    return path


def evaluate(model, device, ref_dir, query_dir):
    net = unwrap(model).eval()
    ref = eval_utils.load_ref_img(net, device, ref_dir)
    recall_results, scores = eval_utils.recall(net, device, ref, query_dir)
    return scores, recall_results


def train_epoch(model, loader, optimizer, device, scaler, beta, margin, log_interval):
    model.train()
    running_loss = 0.0
    for step, batch in enumerate(loader, 1):
        anchors, positives, negatives, _ = batch
        anchors = anchors.to(device, non_blocking=True)
        positives = positives.to(device, non_blocking=True)
        negatives = negatives.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            ga, sa = forward_descriptors(model, anchors)
            gp, sp = forward_descriptors(model, positives)
            gn, sn = forward_descriptors(model, negatives)

            loss_global = cosine_triplet_loss(ga, gp, gn, margin)
            loss_semi = cosine_triplet_loss(sa, sp, sn, margin)
            loss = beta * loss_global + (1.0 - beta) * loss_semi

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        running_loss += loss.item()
        if log_interval and step % log_interval == 0:
            avg = running_loss / log_interval
            print(f"[train] step={step:05d} loss={avg:.4f}")
            running_loss = 0.0
    return running_loss


def parse_args():
    parser = argparse.ArgumentParser(description="Train SGM-Net")
    parser.add_argument("--dataset-root", required=True, help="Root folder of University-Release.")
    parser.add_argument("--train-sat-subdir", default="train/gallery_satellite")
    parser.add_argument("--train-drone-subdir", default="train/query_drone")
    parser.add_argument("--val-sat-subdir", default="val/gallery_satellite")
    parser.add_argument("--val-drone-subdir", default="val/query_drone")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--steps-per-epoch", type=int, default=2000)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--margin", type=float, default=0.2)
    parser.add_argument("--beta", type=float, default=0.5, help="Weight on global loss.")
    parser.add_argument("--alpha", type=float, default=1.0, help="CSMG scaling factor.")
    parser.add_argument("--num-clusters", type=int, default=4)
    parser.add_argument("--backbone", choices=["vgg16", "resnet50"], default="vgg16")
    parser.add_argument("--no-freeze-backbone", action="store_true")
    parser.add_argument("--mixed-precision", action="store_true")
    parser.add_argument("--eval-interval", type=int, default=5)
    parser.add_argument("--output-dir", default="./checkpoints")
    parser.add_argument("--resume", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=20)
    return parser.parse_args()


def main():
    args = parse_args()
    set_seed(args.seed)
    torch.backends.cudnn.benchmark = True

    train_tf, eval_tf = build_transforms()

    train_sat = os.path.join(args.dataset_root, args.train_sat_subdir)
    train_drone = os.path.join(args.dataset_root, args.train_drone_subdir)

    train_dataset = TripletDataset(
        train_sat,
        train_drone,
        transform=train_tf,
        length=args.steps_per_epoch * args.batch_size,
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=8,
        pin_memory=True,
    )

    backbone, out_channels = build_backbone(args.backbone, freeze=not args.no_freeze_backbone)
    graph_head = CSMG(input_channel=out_channels, num_clusters=args.num_clusters, alpha=args.alpha)
    model = JointNet(backbone, graph_head)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    scaler = torch.cuda.amp.GradScaler() if args.mixed_precision and device.type == "cuda" else None
    start_epoch = 0
    metrics_history = []

    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        unwrap(model).load_state_dict(checkpoint["model_state"])
        optimizer.load_state_dict(checkpoint["optim_state"])
        if scaler and "scaler_state" in checkpoint:
            scaler.load_state_dict(checkpoint["scaler_state"])
        start_epoch = checkpoint["epoch"] + 1
        metrics_history = checkpoint.get("history", [])

    print(f"Starting training for {args.epochs} epochs on {device}.")
    for epoch in range(start_epoch, args.epochs):
        tic = time.time()
        residual = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            scaler,
            args.beta,
            args.margin,
            args.log_interval,
        )
        elapsed = time.time() - tic
        print(f"[epoch {epoch+1:03d}] residual_loss={residual:.4f} time={elapsed:.1f}s")

        if (epoch + 1) % args.eval_interval == 0:
            val_sat = os.path.join(args.dataset_root, args.val_sat_subdir)
            val_drone = os.path.join(args.dataset_root, args.val_drone_subdir)
            if os.path.exists(val_sat) and os.path.exists(val_drone):
                scores, _ = evaluate(model, device, val_sat, val_drone)
                print(f"[eval] R@1={scores['R1']:.2f} R@5={scores['R5']:.2f}")
                metrics_history.append({"epoch": epoch + 1, "scores": scores})
            else:
                print("[eval] skipped (validation folders not found).")

        state = {
            "epoch": epoch,
            "model_state": unwrap(model).state_dict(),
            "optim_state": optimizer.state_dict(),
            "scaler_state": scaler.state_dict() if scaler else None,
            "history": metrics_history,
            "args": vars(args),
            "timestamp": time.time(),
        }
        ckpt_path = save_checkpoint(state, args.output_dir, f"sgm_net_epoch_{epoch+1:03d}")
        print(f"[checkpoint] saved to {ckpt_path}")

    hist_path = os.path.join(args.output_dir, "training_history.json")
    with open(hist_path, "w", encoding="utf-8") as f:
        json.dump(metrics_history, f, indent=2)
    print(f"[done] history saved to {hist_path}")


if __name__ == "__main__":
    main()
