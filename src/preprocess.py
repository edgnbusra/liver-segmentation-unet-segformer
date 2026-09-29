"""LiTS NIfTI volume'larini 2D dilimlere ayirip .npy olarak diske kaydeder.

Neden bu adima ihtiyacimiz var?
- Her CT volume'u (512, 512, 75) gibi 3 boyutlu ve nibabel ile okunmasi
  goreceli yavas. Egitim sirasinda her epoch'ta ayni volume'lari tekrar
  tekrar NIfTI'den okumak yerine, tek seferde 2D dilimlere bolup hafif
  .npy dosyalari olarak kaydediyoruz. Boylece:
    1) Egitim dongusu cok daha hizli calisir (diskten hazir array okumak,
       NIfTI parse etmekten cok daha ucuz).
    2) Colab'a tasimasi kolaylasir (islenmis .npy'lari yuklemek yeterli).
"""

import os
import glob

import cv2
import numpy as np
import nibabel as nib

# ---- Ayarlar ----
RAW_VOLUME_DIR = "data/raw/extracted/volume_pt1"
RAW_SEG_DIR = "data/raw/extracted/segmentations"
OUT_IMAGE_DIR = "data/processed/images"
OUT_MASK_DIR = "data/processed/masks"

IMG_SIZE = 256          # egitimde kullanilacak 2D goruntu boyutu
HU_MIN, HU_MAX = -200, 250   # "karaciger penceresi" - bu HU araligi disindaki
                              # dokular (kemik, hava) karaciger ayrimi icin
                              # onemli degil, bu yuzden kirpiyoruz (clipping)


def normalize_hu(slice_2d):
    """CT slice'ini [HU_MIN, HU_MAX] araligina kirpip [0, 1]'e olcekler."""
    slice_2d = np.clip(slice_2d, HU_MIN, HU_MAX)
    slice_2d = (slice_2d - HU_MIN) / (HU_MAX - HU_MIN)
    return slice_2d.astype(np.float32)


def binarize_liver_mask(mask_2d):
    """Label 1 (karaciger) ve 2 (tumor) degerlerini birlestirip
    tek bir 'karaciger var/yok' binary maskesi olusturur."""
    return (mask_2d > 0).astype(np.float32)


def process_volume(volume_path, seg_path, volume_id):
    vol = nib.load(volume_path).get_fdata()   # (H, W, num_slices)
    seg = nib.load(seg_path).get_fdata()

    num_slices = vol.shape[2]
    saved = 0

    for i in range(num_slices):
        img_slice = vol[:, :, i]
        mask_slice = seg[:, :, i]

        img_slice = normalize_hu(img_slice)
        mask_slice = binarize_liver_mask(mask_slice)

        img_slice = cv2.resize(img_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_LINEAR)
        # Maske icin INTER_NEAREST kullaniyoruz: interpolasyon 0/1 disinda
        # ara degerler (0.3, 0.7 gibi) uretmesin, maske hep 0 veya 1 kalsin.
        mask_slice = cv2.resize(mask_slice, (IMG_SIZE, IMG_SIZE), interpolation=cv2.INTER_NEAREST)

        out_name = f"vol{volume_id}_slice{i:03d}.npy"
        np.save(os.path.join(OUT_IMAGE_DIR, out_name), img_slice)
        np.save(os.path.join(OUT_MASK_DIR, out_name), mask_slice)
        saved += 1

    return saved


def main():
    os.makedirs(OUT_IMAGE_DIR, exist_ok=True)
    os.makedirs(OUT_MASK_DIR, exist_ok=True)

    volume_paths = sorted(
        glob.glob(os.path.join(RAW_VOLUME_DIR, "volume-*.nii")),
        key=lambda p: int(os.path.basename(p).replace("volume-", "").replace(".nii", "")),
    )

    total_slices = 0
    for vpath in volume_paths:
        volume_id = os.path.basename(vpath).replace("volume-", "").replace(".nii", "")
        spath = os.path.join(RAW_SEG_DIR, f"segmentation-{volume_id}.nii")

        if not os.path.exists(spath):
            print(f"[UYARI] segmentation-{volume_id}.nii bulunamadi, atlaniyor.")
            continue

        n = process_volume(vpath, spath, volume_id)
        total_slices += n
        print(f"volume-{volume_id}: {n} dilim islendi.")

    print(f"\nToplam {total_slices} 2D dilim '{OUT_IMAGE_DIR}' ve '{OUT_MASK_DIR}' altina kaydedildi.")


if __name__ == "__main__":
    main()
