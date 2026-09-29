"""Hazir (pretrained) SegFormer'i bizim binary karaciger segmentasyonu
gorevimize uyarlayan sarmalayici (wrapper).

SegFormer normalde:
  - 3 kanal (RGB) girdi bekler          -> biz 1 kanal (gri) griyi 3'e kopyalayarak cozeriz
  - N farkli sinif icin egitilmistir     -> biz num_labels=1 ile son katmani degistiririz
  - Cikti, girdinin 1/4 cozunurlugunde  -> biz interpolate ile geri buyuturuz

Boylece disaridan bakildiginda, bu model tipki bizim UNet sinifimiz gibi
davranir: [B, 1, H, W] girdi alir, [B, 1, H, W] logit dondurur - ayni
DiceBCELoss ve iou_score fonksiyonlariyla dogrudan kullanilabilir.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import SegformerForSemanticSegmentation

MODEL_NAME = "nvidia/segformer-b0-finetuned-ade-512-512"


class SegFormerBinary(nn.Module):
    def __init__(self, model_name=MODEL_NAME):
        super().__init__()
        # ignore_mismatched_sizes=True: HuggingFace'e "son siniflandirma
        # katmaninin boyutu uyusmuyor, onu at ve num_labels'a gore yenisini
        # rastgele baslat" diyoruz. Geri kalan tum agirliklar (encoder)
        # onceden egitilmis haliyle yuklenir.
        self.segformer = SegformerForSemanticSegmentation.from_pretrained(
            model_name,
            num_labels=1,
            ignore_mismatched_sizes=True,
        )

    def forward(self, x):
        # x: [B, 1, H, W] (bizim gri tonlama CT dilimlerimiz)
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)  # [B, 1, H, W] -> [B, 3, H, W]

        target_size = x.shape[2:]  # orijinal H, W - cikisi buna geri buyutecegiz

        outputs = self.segformer(pixel_values=x)
        logits = outputs.logits  # [B, 1, H/4, W/4]

        logits = F.interpolate(logits, size=target_size, mode="bilinear", align_corners=False)
        return logits


if __name__ == "__main__":
    # Sanity check: UNet testimizle birebir ayni sekilde, rastgele bir
    # goruntu ile boyutlarin dogru aktigini dogruluyoruz.
    model = SegFormerBinary()
    dummy_input = torch.randn(2, 1, 256, 256)
    output = model(dummy_input)
    print("Girdi boyutu :", dummy_input.shape)
    print("Cikti boyutu :", output.shape)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Toplam parametre sayisi: {n_params:,}")
