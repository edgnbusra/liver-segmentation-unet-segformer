"""Dunku Colab egitiminin (U-Net baseline) sonuclarini geriye donuk olarak
MLflow'a isler.

Bu egitim, MLflow entegrasyonunu train.py'ye eklemeden ONCE Colab'da
calistirilmisti, bu yuzden o an otomatik loglanmadi. Ama elimizde tam epoch
loglari (chat'ten kopyalanan) ve duzeltilmis nihai Dice/IoU degerleri
(evaluate.py ciktisi) oldugu icin, bunlari MLflow'a "sanki o an loglanmis
gibi" isleyebiliyoruz. Boylece SegFormer ile karsilastirma yaparken U-Net
baseline'i da ayni tabloda hazır olacak.

Kullanim:
    python log_baseline_to_mlflow.py
"""

import mlflow

EPOCH_LOGS = [
    (1, 1.1585, 1.0846, 0.7596),
    (2, 0.9358, 0.9578, 0.9504),
    (3, 0.7625, 0.8618, 0.9021),
    (4, 0.5766, 0.7940, 0.9500),
    (5, 0.3981, 0.7467, 0.9568),
    (6, 0.2665, 0.7528, 0.8853),
    (7, 0.1850, 0.7269, 0.3949),
    (8, 0.1353, 0.6865, 0.9582),
    (9, 0.1030, 0.7224, 0.9117),
    (10, 0.0877, 0.6783, 0.9681),
    (11, 0.0642, 0.6713, 0.9755),
    (12, 0.0567, 0.6693, 0.9737),
    (13, 0.0500, 0.6668, 0.9724),
    (14, 0.0473, 0.6665, 0.9708),
    (15, 0.0432, 0.6739, 0.9284),
    (16, 0.0428, 0.6628, 0.9732),
    (17, 0.0377, 0.6599, 0.9741),
    (18, 0.0323, 0.6595, 0.9585),
    (19, 0.0282, 0.6566, 0.9618),
    (20, 0.0248, 0.6588, 0.9753),
    (21, 0.0329, 0.6643, 0.9636),
    (22, 0.0372, 0.6581, 0.9715),
    (23, 0.0217, 0.6559, 0.9742),
    (24, 0.0241, 0.6548, 0.9756),
    (25, 0.0306, 0.6535, 0.9778),
]


def main():
    mlflow.set_experiment("liver-segmentation")

    with mlflow.start_run(run_name="unet-baseline"):
        mlflow.log_params({
            "model": "UNet",
            "epochs": 25,
            "batch_size": 16,
            "lr": 1e-4,
            "device": "cuda",
            "train_samples": 4227,
            "val_samples": 1050,
        })

        best_iou = 0.0
        for epoch, train_loss, val_loss, val_iou in EPOCH_LOGS:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }, step=epoch)
            best_iou = max(best_iou, val_iou)

        # evaluate.py'nin duzelttigi, bos dilimlerin yapay olarak sismedigi
        # gercek performans sayilari - SegFormer ile adil karsilastirma
        # bunlar uzerinden yapilacak.
        mlflow.log_metrics({
            "best_val_iou_raw": best_iou,               # egitim sirasinda raporlanan (yaniltici) IoU
            "final_dice_all_slices": 0.9853,
            "final_iou_all_slices": 0.9767,
            "final_dice_liver_only": 0.9565,
            "final_iou_liver_only": 0.9308,
            "final_dice_aggregate": 0.9766,
            "final_iou_aggregate": 0.9543,
        })

        mlflow.log_artifact("checkpoints/best_model.pth")
        mlflow.log_artifact("results/predictions.png")

        print("Baseline run MLflow'a kaydedildi.")


if __name__ == "__main__":
    main()
