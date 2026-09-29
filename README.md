# U-Net ile Medikal Görüntü Segmentasyonu (LiTS Dataset)

Bu proje, U-Net encoder-decoder mimarisini sıfırdan kullanarak LiTS
(Liver Tumor Segmentation Challenge) veri setinde karaciğer
segmentasyonu yapmayı amaçlar.

## Klasör Yapısı

- `data/raw/` — LiTS'ten indirilen ham NIfTI (.nii/.nii.gz) veriler buraya konacak
- `data/processed/` — Ön işlenmiş (NIfTI/npy) veriler
- `src/` — Kaynak kod (dataset okuma, model, loss, metrikler, eğitim döngüsü)
- `notebooks/` — Colab/Jupyter notebook'ları
- `checkpoints/` — Eğitilmiş model ağırlıkları
- `results/` — Tahmin görselleri, metrik grafikleri

## Plan

1. [x] Proje iskeleti
2. [ ] LiTS dataset indirme (Kaggle: "LiTS - Liver Tumor Segmentation")
3. [ ] Veri okuma & ön işleme (NIfTI)
4. [ ] U-Net modeli (PyTorch, skip connections)
5. [ ] Dice + BCE combined loss, IoU metriği
6. [ ] Colab üzerinde eğitim
