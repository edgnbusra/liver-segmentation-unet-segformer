"""Bir CT volume'unun tum dilimlerini gezip, gercek maske + model tahmini
overlay'i ile birlikte bir GIF animasyonu uretir.

Neden buna ihtiyacimiz var?
Tek bir 2D dilime bakmak, bir 3D yapinin (karacigerin) sadece TEK bir kesitini
gormek demek. Bir CT taramasi onlarca/yuzlerce dilimden olustugu icin, tum
dilimleri sirayla oynatarak karacigerin 3 boyuttaki gercek seklini (nerede
basliyor, nerede genisliyor, nerede bitiyor) gozle gorebiliyoruz. Bu, modelin
sadece tek bir "sansli" kesitte degil, TUM tarama boyunca tutarli calistigini
da gosterir.

Kullanim:
    python make_3d_gif.py --volume_id 9 --checkpoint checkpoints/best_model.pth
"""

import argparse
import os

import cv2
import imageio
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

from model import UNet

HU_MIN, HU_MAX = -200, 250
IMG_SIZE = 256


def normalize_hu(slice_2d):
    slice_2d = np.clip(slice_2d, HU_MIN, HU_MAX)
    return ((slice_2d - HU_MIN) / (HU_MAX - HU_MIN)).astype(np.float32)


def make_overlay_frame(ct_slice, gt_mask, pred_mask):
    """Tek bir dilim icin: CT gri tonlama uzerine gercek maske (yesil) ve
    tahmin maskesi (kirmizi) yari saydam bindirilmis bir RGB kare uretir.

    Neden iki farkli renk? Yesil ve kirmizinin UST USTE bindigi (ikisi de
    aynen hemfikir oldugu) bolgeler SARI gorunur - gozle "model nerede
    dogru, nerede yanlis" ayrimini hemen yapabiliriz.
    """
    base = (ct_slice * 255).astype(np.uint8)
    rgb = cv2.cvtColor(base, cv2.COLOR_GRAY2RGB).astype(np.float32)

    overlay = rgb.copy()
    overlay[gt_mask > 0.5] = overlay[gt_mask > 0.5] * 0.4 + np.array([0, 255, 0]) * 0.6
    overlay[pred_mask > 0.5] = overlay[pred_mask > 0.5] * 0.4 + np.array([255, 0, 0]) * 0.6

    return overlay.astype(np.uint8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--volume_id", type=str, default="9",
                         help="Hangi volume gosterilecek (validation setinden 9 veya 10 onerilir)")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/best_model.pth")
    parser.add_argument("--volume_dir", type=str, default="data/raw/extracted/volume_pt1")
    parser.add_argument("--seg_dir", type=str, default="data/raw/extracted/segmentations")
    parser.add_argument("--out_path", type=str, default="results/volume_animation.gif")
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--padding", type=int, default=5,
                         help="Karacigerin gorundugu araligin oncesine/sonrasina "
                              "eklenecek ekstra dilim sayisi - karacigerin nasil "
                              "'belirip kayboldugunu' da gormek icin.")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = UNet(in_channels=1, out_channels=1).to(device)
    model.load_state_dict(torch.load(args.checkpoint, map_location=device))
    model.eval()

    vol = nib.load(os.path.join(args.volume_dir, f"volume-{args.volume_id}.nii")).get_fdata()
    seg = nib.load(os.path.join(args.seg_dir, f"segmentation-{args.volume_id}.nii")).get_fdata()

    num_slices = vol.shape[2]

    # Karacigerin GERCEKTEN gorundugu dilim araligini buluyoruz - 549
    # dilimin tamamini degil, sadece anlamli olan kismini gosterecegiz.
    # Boylece hem GIF kucuk kalir hem de izleyici karaciger disindaki
    # (bacak, akciger gibi) alakasiz dilimlerle vakit kaybetmez.
    liver_slice_indices = [i for i in range(num_slices) if (seg[:, :, i] > 0).sum() > 0]
    if not liver_slice_indices:
        raise ValueError(f"volume-{args.volume_id} icinde hic karaciger pikseli bulunamadi.")

    start = max(0, min(liver_slice_indices) - args.padding)
    end = min(num_slices - 1, max(liver_slice_indices) + args.padding)
    slice_range = range(start, end + 1)

    print(f"volume-{args.volume_id}: toplam {num_slices} dilim, "
          f"karaciger {min(liver_slice_indices)}-{max(liver_slice_indices)} araliginda, "
          f"gosterilecek aralik: {start}-{end} ({len(slice_range)} dilim)")

    frames = []
    with torch.no_grad():
        for i in slice_range:
            ct_slice = normalize_hu(vol[:, :, i])
            gt_slice = (seg[:, :, i] > 0).astype(np.float32)

            ct_resized = cv2.resize(ct_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
            gt_resized = cv2.resize(gt_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

            tensor = torch.from_numpy(ct_resized).unsqueeze(0).unsqueeze(0).float().to(device)
            pred = (torch.sigmoid(model(tensor)) > 0.5).float().squeeze().cpu().numpy()

            frame = make_overlay_frame(ct_resized, gt_resized, pred)

            # Uzerine dilim numarasini yaziyoruz ki GIF oynarken "hangi
            # kesitteyiz" takip edilebilsin. Kucuk figsize+dpi: dosya
            # boyutunu makul tutmak icin (549 tam-boy kareyle 33MB'a
            # cikmisti, kucultulmus araliğa ve dpi'ye gecince cok daha
            # kucuk kalir).
            fig, ax = plt.subplots(figsize=(3, 3), dpi=80)
            ax.imshow(frame)
            ax.set_title(f"volume-{args.volume_id} | dilim {i+1}/{num_slices}\n"
                         f"yesil=gercek, kirmizi=tahmin, sari=ikisi de dogru", fontsize=6)
            ax.axis("off")
            fig.tight_layout()

            fig.canvas.draw()
            buf = np.asarray(fig.canvas.buffer_rgba())
            frames.append(buf[:, :, :3].copy())
            plt.close(fig)

    os.makedirs(os.path.dirname(args.out_path), exist_ok=True)
    imageio.mimsave(args.out_path, frames, fps=args.fps)
    print(f"GIF kaydedildi: {args.out_path} ({len(frames)} kare)")


if __name__ == "__main__":
    main()
