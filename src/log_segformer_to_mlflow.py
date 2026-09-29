"""SegFormer'in Colab'daki egitim ciktisini (epoch loglarini) yerel MLflow'a
isler - tipki log_baseline_to_mlflow.py'nin U-Net icin yaptigi gibi.

Colab'dan gelen "Epoch X/15 | train_loss=... | val_loss=... | val_IoU=..."
satirlarini asagidaki EPOCH_LOGS listesine ekleyip calistir.

Kullanim:
    python log_segformer_to_mlflow.py
"""

import mlflow

EPOCH_LOGS = [
    (1, 0.9623, 0.8412, 0.8710),
    (2, 0.3795, 0.7268, 0.9133),
    (3, 0.1943, 0.7115, 0.9191),
    (4, 0.1507, 0.7013, 0.9347),
    (5, 0.1246, 0.6980, 0.9137),
    (6, 0.1004, 0.6954, 0.9082),
    (7, 0.0948, 0.6906, 0.9353),
    (8, 0.0978, 0.6929, 0.9328),
    (9, 0.1023, 0.6881, 0.9448),
    (10, 0.0870, 0.6877, 0.9376),
    (11, 0.0880, 0.6870, 0.9394),
    (12, 0.0906, 0.6863, 0.9477),
    (13, 0.0882, 0.6959, 0.9380),
    (14, 0.0892, 0.6846, 0.9416),
    (15, 0.0853, 0.6855, 0.9406),
]

BEST_VAL_IOU = 0.9477


def main():
    mlflow.set_experiment("liver-segmentation")

    with mlflow.start_run(run_name="segformer-b0"):
        mlflow.log_params({
            "model": "SegFormer",
            "pretrained_checkpoint": "nvidia/segformer-b0-finetuned-ade-512-512",
            "epochs": len(EPOCH_LOGS),
            "batch_size": 8,
            "lr": 6e-5,
            "device": "cuda",
            "train_samples": 4227,
            "val_samples": 1050,
            "total_params": 3_714_401,
        })

        best_iou = 0.0
        for epoch, train_loss, val_loss, val_iou in EPOCH_LOGS:
            mlflow.log_metrics({
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_iou": val_iou,
            }, step=epoch)
            best_iou = max(best_iou, val_iou)

        mlflow.log_metric("best_val_iou", best_iou)

        # evaluate_segformer.py'nin duzelttigi gercek performans sayilari -
        # U-Net'in log_baseline_to_mlflow.py'deki final_ metrikleriyle
        # birebir ayni yontemle hesaplanmis, dogrudan karsilastirilabilir.
        mlflow.log_metrics({
            "final_dice_all_slices": 0.9750,
            "final_iou_all_slices": 0.9628,
            "final_dice_liver_only": 0.9259,
            "final_iou_liver_only": 0.8898,
            "final_dice_aggregate": 0.9666,
            "final_iou_aggregate": 0.9354,
        })

        import os
        if os.path.exists("checkpoints/segformer_best.pth"):
            mlflow.log_artifact("checkpoints/segformer_best.pth")

        print("SegFormer run MLflow'a kaydedildi.")


if __name__ == "__main__":
    main()
