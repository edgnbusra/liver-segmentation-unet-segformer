"""Egitilmis model icin daha dogru bir degerlendirme yapar.

train.py'deki val_IoU, TUM dilimlerin (bos olanlar dahil) ortalamasiydi.
Validation setinin ~%85'i karaciger icermeyen bos dilim oldugu icin, model
bu dilimlerde "hicbir sey yok" deyip kolayca IoU=1 aldigindan, ortalama
skor gercekte oldugundan cok daha iyi gorunuyordu.

Burada UC farkli sekilde raporluyoruz:

1. all_slices  : train.py'deki ile ayni yontem (referans icin, tum dilimler)
2. liver_only  : SADECE gercekte karaciger iceren dilimler uzerinden
                 ortalama (bos dilimleri tamamen disariya atarak)
3. aggregate   : Tum validation setindeki piksel kesisim/birlesimlerini
                 TOPLAYIP tek bir Dice/IoU hesabi yapmak (literatede en
                 yaygin kullanilan, "case-wise" degil "global" yontem).
                 Bos dilimler kesisim=0, birlesim=0 katkisi yaptigi icin
                 sonucu yapay olarak sismiyorlar, ama tek bir zor ornegin
                 etkisini de aci tarafa cekmiyor.

Kullanim:
    python evaluate.py --checkpoint checkpoints/best_model.pth
"""

import argparse

import torch

from model import UNet
from dataset import get_dataloaders


def dice_iou_from_counts(intersection, pred_sum, target_sum, smooth=1e-6):
    union = pred_sum + target_sum - intersection
    dice = (2 * intersection + smooth) / (pred_sum + target_sum + smooth)
    iou = (intersection + smooth) / (union + smooth)
    return dice, iou


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, val_loader = get_dataloaders(batch_size=8, num_workers=0)

    all_slice_dices, all_slice_ious = [], []
    liver_only_dices, liver_only_ious = [], []
    total_intersection, total_pred_sum, total_target_sum = 0.0, 0.0, 0.0

    with torch.no_grad():
        for imgs, masks in val_loader:
            imgs, masks = imgs.to(device), masks.to(device)
            preds = (torch.sigmoid(model(imgs)) > args.threshold).float()

            for i in range(imgs.size(0)):
                p = preds[i].view(-1)
                t = masks[i].view(-1)

                intersection = (p * t).sum().item()
                pred_sum = p.sum().item()
                target_sum = t.sum().item()

                dice, iou = dice_iou_from_counts(intersection, pred_sum, target_sum)
                all_slice_dices.append(dice)
                all_slice_ious.append(iou)

                if target_sum > 0:  # sadece gercekte karaciger olan dilimler
                    liver_only_dices.append(dice)
                    liver_only_ious.append(iou)

                total_intersection += intersection
                total_pred_sum += pred_sum
                total_target_sum += target_sum

    agg_dice, agg_iou = dice_iou_from_counts(total_intersection, total_pred_sum, total_target_sum)

    def avg(lst):
        return sum(lst) / len(lst) if lst else float("nan")

    print(f"Toplam validation dilim sayisi     : {len(all_slice_dices)}")
    print(f"Karaciger iceren dilim sayisi       : {len(liver_only_dices)}\n")

    print("1) Tum dilimler (bos dilimler dahil, train.py'deki gibi):")
    print(f"   Dice = {avg(all_slice_dices):.4f} | IoU = {avg(all_slice_ious):.4f}\n")

    print("2) Sadece karaciger iceren dilimler:")
    print(f"   Dice = {avg(liver_only_dices):.4f} | IoU = {avg(liver_only_ious):.4f}\n")

    print("3) Agregat (global piksel kesisim/birlesimi, literatur standardi):")
    print(f"   Dice = {agg_dice:.4f} | IoU = {agg_iou:.4f}")


if __name__ == "__main__":
    main()
