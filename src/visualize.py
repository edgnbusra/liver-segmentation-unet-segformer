"""Egitilmis modelin birkac validation ornegi uzerindeki tahminlerini
gorsellestirir: CT dilimi | gercek maske | tahmin maskesi yan yana.

Kullanim (Colab'da):
    python visualize.py --checkpoint /content/drive/MyDrive/u_net_checkpoints/best_model.pth --num_samples 6
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch

from model import UNet
from dataset import get_dataloaders


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--num_samples", type=int, default=6,
                         help="Kac ornek gosterilecek")
    parser.add_argument("--only_with_liver", action="store_true", default=True,
                         help="Sadece gercekte karaciger iceren dilimleri sec "
                              "(bos dilimler gorsel acidan ilginc degil)")
    parser.add_argument("--out_path", type=str, default="results/predictions.png")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    _, val_loader = get_dataloaders(batch_size=1, num_workers=0)
    val_dataset = val_loader.dataset

    # Sadece karaciger iceren dilimleri bulup rastgele secim yapiyoruz,
    # boylece gorsellerde bos (tamamen arka plan) dilimler yerine anlamli
    # segmentasyon ornekleri goruruz.
    candidate_indices = []
    for i in range(len(val_dataset)):
        _, mask = val_dataset[i]
        if not args.only_with_liver or mask.sum() > 0:
            candidate_indices.append(i)
        if len(candidate_indices) >= args.num_samples * 5:
            break  # tum veri setini taramaya gerek yok, yeterince aday bulunca dur

    chosen = np.random.choice(candidate_indices, size=min(args.num_samples, len(candidate_indices)), replace=False)

    fig, axes = plt.subplots(len(chosen), 3, figsize=(9, 3 * len(chosen)))
    if len(chosen) == 1:
        axes = axes[np.newaxis, :]

    for row, idx in enumerate(chosen):
        img, mask = val_dataset[idx]
        img_batch = img.unsqueeze(0).to(device)

        with torch.no_grad():
            logits = model(img_batch)
            pred = (torch.sigmoid(logits) > 0.5).float()

        img_np = img.squeeze().numpy()
        mask_np = mask.squeeze().numpy()
        pred_np = pred.squeeze().cpu().numpy()

        axes[row, 0].imshow(img_np, cmap="gray")
        axes[row, 0].set_title("CT dilimi")
        axes[row, 1].imshow(mask_np, cmap="gray")
        axes[row, 1].set_title("Gercek maske")
        axes[row, 2].imshow(pred_np, cmap="gray")
        axes[row, 2].set_title("Model tahmini")

        for ax in axes[row]:
            ax.axis("off")

    plt.tight_layout()
    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    plt.savefig(args.out_path, dpi=150)
    print(f"Gorsel kaydedildi: {args.out_path}")


if __name__ == "__main__":
    main()
