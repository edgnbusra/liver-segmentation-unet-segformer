"""U-Net modeli (PyTorch) - sifirdan, skip connection'lar dahil."""

import torch
import torch.nn as nn


class DoubleConv(nn.Module):
    """(Conv2d -> BatchNorm -> ReLU) iki kez ust uste.

    U-Net'in her seviyesinde (hem encoder hem decoder tarafinda) tekrar
    eden temel blok budur.
    """

    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class UNet(nn.Module):
    """Klasik U-Net: 4 seviyeli encoder, bottleneck, 4 seviyeli decoder.

    in_channels: girdi goruntusunun kanal sayisi (CT icin genelde 1 - gri tonlama)
    out_channels: cikti maskesinin kanal sayisi (binary segmentasyon icin 1)
    features: her encoder seviyesindeki kanal sayilari
    """

    def __init__(self, in_channels=1, out_channels=1, features=(64, 128, 256, 512)):
        super().__init__()

        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # ---- Encoder (daralan yol) ----
        self.encoders = nn.ModuleList()
        ch = in_channels
        for f in features:
            self.encoders.append(DoubleConv(ch, f))
            ch = f

        # ---- Bottleneck (en alt nokta) ----
        self.bottleneck = DoubleConv(features[-1], features[-1] * 2)

        # ---- Decoder (genisleyen yol) ----
        self.upconvs = nn.ModuleList()
        self.decoders = nn.ModuleList()
        rev_features = list(reversed(features))
        in_ch = features[-1] * 2
        for f in rev_features:
            self.upconvs.append(
                nn.ConvTranspose2d(in_ch, f, kernel_size=2, stride=2)
            )
            # concat sonrasi kanal sayisi 2*f olur (upconv ciktisi + skip)
            self.decoders.append(DoubleConv(f * 2, f))
            in_ch = f

        # ---- Cikti katmani ----
        self.final_conv = nn.Conv2d(features[0], out_channels, kernel_size=1)

    def forward(self, x):
        skip_connections = []

        # Encoder: her seviyede conv uygula, sonucu sakla (skip), sonra kucult
        for encoder in self.encoders:
            x = encoder(x)
            skip_connections.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)

        # Skip'leri decoder sirasina gore kullanmak icin ters cevir
        skip_connections = skip_connections[::-1]

        # Decoder: buyut, ayni seviyedeki skip ile birlestir (concat), conv uygula
        for i in range(len(self.decoders)):
            x = self.upconvs[i](x)
            skip = skip_connections[i]

            # Boyut uyuşmazligi olursa (tek sayili genislik/yukseklik gibi
            # durumlarda cikabilir) skip'e gore kirp/hizala
            if x.shape != skip.shape:
                x = nn.functional.interpolate(x, size=skip.shape[2:])

            x = torch.cat([skip, x], dim=1)  # kanal boyutunda birlestir
            x = self.decoders[i](x)

        return self.final_conv(x)


if __name__ == "__main__":
    # Hizli bir sanity check: rastgele bir goruntu ile boyutlarin dogru aktigini test et
    model = UNet(in_channels=1, out_channels=1)
    dummy_input = torch.randn(2, 1, 256, 256)  # (batch, channel, height, width)
    output = model(dummy_input)
    print("Girdi boyutu :", dummy_input.shape)
    print("Cikti boyutu :", output.shape)
