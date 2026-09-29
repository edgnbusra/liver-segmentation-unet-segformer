"""U-Net egitim dongusu.

Kullanim:
    python src/train.py --epochs 20 --batch_size 8
    python src/train.py --epochs 1 --limit 100   # hizli sanity check icin
"""

import argparse
import os
import time

import mlflow
import torch
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

from model import UNet
from losses import DiceBCELoss, iou_score
from dataset import get_dataloaders


def train_one_epoch(model, loader, criterion, optimizer, device, scaler):
    model.train()  # BatchNorm/Dropout gibi katmanlari "egitim modu"na alir
    running_loss = 0.0

    # tqdm: her batch islendikce ilerleme cubugu ve anlik hizi (it/s) gosterir,
    # boylece egitimin gercekten ilerledigini/ne kadar surdugunu canli goruruz.
    pbar = tqdm(loader, desc="train", leave=False)
    for imgs, masks in pbar:
        imgs = imgs.to(device, non_blocking=True)
        masks = masks.to(device, non_blocking=True)

        optimizer.zero_grad()          # bir onceki adimin gradyanlarini temizle

        # Mixed precision (AMP): bazi islemleri float16 ile yaparak GPU'da
        # hem hafiza hem hiz kazandirir, dogruluk kaybi ihmal edilebilir
        # duzeydedir. Sadece GPU varsa (cuda) anlamli, CPU'da etkisiz kalir.
        with torch.autocast(device_type=device.type, enabled=(device.type == "cuda")):
            outputs = model(imgs)      # ileri gecis (forward pass) - tahmin uret
            loss = criterion(outputs, masks)

        scaler.scale(loss).backward()  # geri yayilim (backpropagation)
        scaler.step(optimizer)         # agirliklari guncelle
        scaler.update()

        running_loss += loss.item() * imgs.size(0)
        pbar.set_postfix(loss=loss.item())

    return running_loss / len(loader.dataset)


def validate(model, loader, criterion, device):
    model.eval()  # BatchNorm/Dropout "degerlendirme modu"na gecer (farkli davranirlar)
    running_loss = 0.0
    running_iou = 0.0

    with torch.no_grad():  # val sirasinda gradyan hesaplamaya gerek yok, bellek/hiz kazanci
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
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--limit", type=int, default=None,
                         help="Sadece ilk N ornekle calis (hizli test icin)")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--num_workers", type=int, default=None,
                         help="Veri yukleme icin paralel process sayisi. "
                              "Belirtilmezse GPU'da 2, CPU'da 0 kullanilir "
                              "(Windows'ta CPU + coklu worker overhead'i "
                              "faydadan fazla olabiliyor).")
    parser.add_argument("--experiment_name", type=str, default="liver-segmentation",
                         help="MLflow deney (experiment) adi - ayni deney altindaki "
                              "tum run'lar MLflow arayuzunde birlikte listelenir.")
    parser.add_argument("--run_name", type=str, default="unet-baseline",
                         help="Bu calistirmayi tanimlayan isim (orn. 'unet-baseline', "
                              "'segformer-lr1e-4'). MLflow arayuzunde run'lari "
                              "birbirinden ayirt etmek icin kullanilir.")
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
        # Sanity check icin veri setini kucultuyoruz (gercek egitimde kullanilmaz)
        train_loader = DataLoader(
            Subset(train_loader.dataset, range(min(args.limit, len(train_loader.dataset)))),
            batch_size=args.batch_size, shuffle=True,
        )
        val_loader = DataLoader(
            Subset(val_loader.dataset, range(min(args.limit, len(val_loader.dataset)))),
            batch_size=args.batch_size, shuffle=False,
        )

    model = UNet(in_channels=1, out_channels=1).to(device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    # GradScaler: mixed precision sirasinda float16'nin cok kucuk gradyanlari
    # sifira yuvarlamasini (underflow) onlemek icin loss'u gecici olarak
    # buyutup sonra geri kucultur. Sadece GPU'da anlamli.
    scaler = torch.amp.GradScaler(enabled=(device.type == "cuda"))

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    best_iou = 0.0

    # MLflow: ayni deney (experiment) altinda, bu calistirmayi ayri bir "run"
    # olarak kaydediyoruz. set_experiment, boyle bir deney yoksa olusturur,
    # varsa ona baglanir - boylece U-Net ve SegFormer run'lari ayni tabloda
    # yan yana karşılaştırılabilir hale gelir.
    mlflow.set_experiment(args.experiment_name)
    with mlflow.start_run(run_name=args.run_name):
        # log_param: egitim boyunca DEGISMEYEN ayarlar. Bunlari kaydetmezsek,
        # ileride "bu run hangi learning rate ile egitilmisti?" sorusuna
        # cevap veremeyiz.
        mlflow.log_params({
            "model": "UNet",
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "lr": args.lr,
            "device": device.type,
            "train_samples": len(train_loader.dataset),
            "val_samples": len(val_loader.dataset),
        })

        for epoch in range(1, args.epochs + 1):
            start = time.time()
            train_loss = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler)
            val_loss, val_iou = validate(model, val_loader, criterion, device)
            elapsed = time.time() - start

            print(f"Epoch {epoch}/{args.epochs} | "
                  f"train_loss={train_loss:.4f} | val_loss={val_loss:.4f} | val_IoU={val_iou:.4f} | "
                  f"sure={elapsed:.1f}s")

            # log_metric: DEGISEN degerler, "step" parametresiyle hangi epoch'a
            # ait oldugunu belirtiyoruz. MLflow bunlari otomatik olarak
            # epoch'a karsi bir grafik cizecek sekilde saklar.
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
                "epoch_seconds": elapsed,
            }, step=epoch)

            # En iyi modeli sakla (val IoU'ya gore) - her epoch sonunda degil,
            # sadece iyilesme oldugunda kaydederek disk israfini onluyoruz.
            if val_iou > best_iou:
                best_iou = val_iou
                torch.save(model.state_dict(), os.path.join(args.checkpoint_dir, "best_model.pth"))
                print(f"  -> yeni en iyi model kaydedildi (val_IoU={val_iou:.4f})")

        print(f"\nEgitim tamamlandi. En iyi val IoU: {best_iou:.4f}")

        # log_metric (step'siz): run'in NIHAI sonucu, tek bir sayi olarak.
        # MLflow arayuzundeki run listesinde bu sutunu gorup run'lari
        # siralayabiliriz.
        mlflow.log_metric("best_val_iou", best_iou)

        # log_artifact: egitilmis model dosyasini bu run'a kalici olarak
        # ekliyoruz. Ileride "bu run'in modelini indir" dedigimizde buradan gelir.
        mlflow.log_artifact(os.path.join(args.checkpoint_dir, "best_model.pth"))


if __name__ == "__main__":
    main()
