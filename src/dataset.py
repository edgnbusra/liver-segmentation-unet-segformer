"""Islenmis 2D dilimler icin PyTorch Dataset ve train/val ayrimi.

ONEMLI: Train/val ayrimini RASTGELE dilim bazinda degil, VOLUME (hasta)
bazinda yapiyoruz. Neden?

Ayni CT taramasindan (ayni hastadan) gelen komsu dilimler birbirine cok
benzer (bitisik kesitler neredeyse ayni goruntu). Eger dilimleri rastgele
karistirip train/val'a bolersek, ayni hastanin bir dilimi train'de, hemen
komsu dilimi val'de olabilir. Bu durumda model val setinde "yeni" bir hasta
gormuyor, aslinda ezberledigi bir hastanin cok benzer bir kesitini goruyor.
Bu da val skorlarinin gercekte oldugundan cok daha iyi cikmasina yol acar
(veri sizintisi / data leakage) ve modelin gercekte ne kadar iyi genelledigi
konusunda bizi yaniltir.

Dogru yontem: hangi hastalarin (volume'larin) train, hangilerinin val
oldugunu ONCE belirleyip, o hastalara ait TUM dilimleri ayni gruba koymak.
"""

import glob
import os

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader

IMAGE_DIR = "data/processed/images"
MASK_DIR = "data/processed/masks"


class LiverSliceDataset(Dataset):
    def __init__(self, filenames, image_dir=IMAGE_DIR, mask_dir=MASK_DIR):
        self.filenames = filenames
        self.image_dir = image_dir
        self.mask_dir = mask_dir

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        fname = self.filenames[idx]
        img = np.load(os.path.join(self.image_dir, fname))
        mask = np.load(os.path.join(self.mask_dir, fname))

        # PyTorch conv katmanlari [channel, height, width] bekler, numpy'da
        # elimizde sadece [height, width] var - basina kanal boyutu ekliyoruz.
        img = torch.from_numpy(img).unsqueeze(0).float()
        mask = torch.from_numpy(mask).unsqueeze(0).float()

        return img, mask


def _volume_id_from_filename(fname):
    # "vol3_slice042.npy" -> "3"
    return fname.split("_")[0].replace("vol", "")


def get_dataloaders(batch_size=8, val_volume_ids=("9", "10"), num_workers=2, pin_memory=False):
    """Tum islemis dosyalari tarar, volume id'sine gore train/val'a boler.

    val_volume_ids: hangi volume'larin dogrulama (validation) icin
    ayrilacagini belirtir. Bu hastalara ait dilimler model egitimi
    sirasinda HIC gorulmez, sadece performansi olcmek icin kullanilir.
    """
    all_files = sorted(os.listdir(IMAGE_DIR))

    train_files = [f for f in all_files if _volume_id_from_filename(f) not in val_volume_ids]
    val_files = [f for f in all_files if _volume_id_from_filename(f) in val_volume_ids]

    train_ds = LiverSliceDataset(train_files)
    val_ds = LiverSliceDataset(val_files)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )

    return train_loader, val_loader


if __name__ == "__main__":
    train_loader, val_loader = get_dataloaders(batch_size=4)
    print(f"Train dilim sayisi: {len(train_loader.dataset)}")
    print(f"Val dilim sayisi  : {len(val_loader.dataset)}")

    imgs, masks = next(iter(train_loader))
    print("Batch goruntu boyutu:", imgs.shape)
    print("Batch maske boyutu  :", masks.shape)
    print("Goruntu deger araligi:", imgs.min().item(), "-", imgs.max().item())
    print("Maske benzersiz degerler:", torch.unique(masks))
