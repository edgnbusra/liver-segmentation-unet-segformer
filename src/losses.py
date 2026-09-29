"""Dice + BCE combined loss ve IoU metrigi."""

import torch
import torch.nn as nn


class DiceLoss(nn.Module):
    """Dice katsayisina dayali loss.

    Dice katsayisi iki maske arasindaki ortusmeyi olcer:
        Dice = (2 * |tahmin ve gercek kesisimi|) / (|tahmin| + |gercek|)

    Dice = 1  -> mukemmel ortusme
    Dice = 0  -> hic ortusme yok

    Loss olarak kullanmak icin (1 - Dice) alinir, boylece optimizasyon
    Dice'i 1'e yaklastirmaya calisir.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        # smooth: paydanin sifir olmasini engelleyen kucuk sabit
        # (hem tahmin hem gercek maske tamamen bos oldugunda 0/0 hatasini onler)
        self.smooth = smooth

    def forward(self, logits, targets):
        # logits: modelin ham ciktisi (sigmoid uygulanmamis), sekil [B, 1, H, W]
        # targets: gercek maske, 0/1 degerli, ayni sekil
        probs = torch.sigmoid(logits)  # once olasiliga cevir (0-1 arasi)

        probs = probs.view(-1)
        targets = targets.view(-1)  # duz vektore cevir, piksel piksel karsilastir

        intersection = (probs * targets).sum()
        dice = (2.0 * intersection + self.smooth) / (
            probs.sum() + targets.sum() + self.smooth
        )
        return 1.0 - dice


class DiceBCELoss(nn.Module):
    """Dice loss + Binary Cross Entropy loss toplami.

    Neden ikisi birden?
    - BCE: her pikseli tek tek, birbirinden bagimsiz degerlendirir. Egitimin
      basinda kararli ve guclu bir gradyan saglar, ama sinif dengesizligine
      (goruntude cok az karaciger pikseli, cok fazla arka plan pikseli varsa)
      duyarlidir; model kolayca "her yeri arka plan de" diyerek dusuk loss
      elde edebilir.
    - Dice: buyuk/kucuk nesne ayrimi yapmadan, ortusme oranina odaklanir. Sinif
      dengesizligine BCE kadar duyarli degildir, kucuk karaciger bolgelerinde
      de anlamli bir sinyal verir.

    Ikisini toplayarak, BCE'nin egitimi stabilize eden etkisini, Dice'in
    ortusmeye odaklanan etkisiyle birlestirmis oluyoruz.
    """

    def __init__(self, smooth=1e-6):
        super().__init__()
        self.dice = DiceLoss(smooth=smooth)
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        return self.dice(logits, targets) + self.bce(logits, targets)


def iou_score(logits, targets, threshold=0.5, smooth=1e-6):
    """Intersection over Union (Jaccard index) metrigi.

    IoU = |tahmin ve gercek kesisimi| / |tahmin ve gercek birlesimi|

    Egitim sirasinda optimize edilen bir loss degil, modelin ne kadar iyi
    oldugunu insanin anlayacagi sekilde raporlamak icin kullanilan bir
    degerlendirme metrigidir. Bu yuzden gradyan hesaplamaya gerek yok
    (torch.no_grad).
    """
    with torch.no_grad():
        probs = torch.sigmoid(logits)
        preds = (probs > threshold).float()  # olasiligi kesin 0/1 maskeye cevir
        targets = targets.float()

        preds = preds.view(-1)
        targets = targets.view(-1)

        intersection = (preds * targets).sum()
        union = preds.sum() + targets.sum() - intersection

        iou = (intersection + smooth) / (union + smooth)
        return iou.item()


if __name__ == "__main__":
    # Sanity check: rastgele tahmin ve gercek maske ile loss/metrik hesapla
    torch.manual_seed(0)
    logits = torch.randn(2, 1, 64, 64)          # modelin ham ciktisi (sigmoid oncesi)
    targets = torch.randint(0, 2, (2, 1, 64, 64)).float()  # rastgele 0/1 gercek maske

    criterion = DiceBCELoss()
    loss = criterion(logits, targets)
    iou = iou_score(logits, targets)

    print("Loss (Dice+BCE):", loss.item())
    print("IoU             :", iou)
