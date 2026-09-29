"""SegFormer fine-tuning dongusu.

train.py ile ayni yapida (ayni loss, metrik, MLflow entegrasyonu) - tek
fark model mimarisi. Boylece U-Net ile SegFormer sonuclari birebir adil
sekilde karsilastirilabilir.

Kullanim:
    python train_segformer.py --epochs 15 --batch_size 8
"""

import argparse
import os
import time

import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from segformer_model import SegFormerBinary, MODEL_NAME
from losses import DiceBCELoss, iou_score
from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()
    running_loss = 0.0

    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(imgs)
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():
        for imgs, masks in tqdm(loader, desc="val", leave=False):
            imgs = imgs.to(device, non_blocking=True)
            masks = masks.to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
                outputs = model(imgs)
                loss = criterion(outputs, masks)

            running_loss += loss.item() * imgs.size(0)
            running_iou += iou_score(outputs, masks) * imgs.size(0)

    n = len(loader.dataset)
    return running_loss / n, running_iou / n


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=15,
                         help="U-Net'ten daha az epoch yeterli olabilir - model "
                              "zaten onceden egitilmis (pretrained), sifirdan "
                              "ogrenmiyor, sadece ince ayar (fine-tune) yapiyor.")
    parser.add_argument("--batch_size", type=int, default=8,
                         help="SegFormer, U-Net'ten daha fazla GPU bellegi "
                              "kullanabilir (transformer katmanlari agir), bu "
                              "yuzden varsayilan batch_size'i dusuk tuttuk.")
    parser.add_argument("--lr", type=float, default=6e-5,
                         help="Pretrained modelleri fine-tune ederken genelde "
                              "sifirdan egitimden DAHA DUSUK bir learning rate "
                              "kullanilir - aksi halde onceden ogrenilmis "
                              "faydali agirliklar hizla bozulabilir "
                              "(catastrophic forgetting).")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=None)
    parser.add_argument("--experiment_name", type=str, default="liver-segmentation")
    parser.add_argument("--run_name", type=str, default="segformer-b0")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Cihaz: {device}")

    num_workers = args.num_workers
    if num_workers is None:
        num_workers = 2 if device.type == "cuda" else 0

    train_loader, val_loader = get_dataloaders(
        batch_size=args.batch_size,
        num_workers=num_workers,
        pin_memory=(device.type == "cuda"),
    )

    if args.limit is not None:
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(min(args.limit, len(train_loader.dataset)))),
            batch_size=args.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(min(args.limit, len(val_loader.dataset)))),
            batch_size=args.batch_size, shuffle=False,
        )

    model = SegFormerBinary().to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_iou = 0.0

    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        mlflow.log_params({
            "model": "SegFormer",
            "pretrained_checkpoint": MODEL_NAME,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": device.type,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
            "total_params": sum(p.numel() for p in model.parameters()),
        })

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_iou = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start

            print(f"Epoch {epoch}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_IoU={val_iou:.4f} | "
                  f"sure={elapsed:.1f}s")

            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "epoch_seconds": elapsed,
            }, step=epoch)

            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "segformer_best.pth"))
                print(f"  -> yeni en iyi model kaydedildi (val_IoU={val_iou:.4f})")

        print(f"\nEgitim tamamlandi. En iyi val IoU: {best_iou:.4f}")
        mlflow.log_metric("best_val_iou", best_iou)
        mlflow.log_artifact(os.path.join(args.checkpoint_dir, "segformer_best.pth"))


if __name__ == "__main__":
    main()
