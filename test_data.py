# test_data.py
import os
from pathlib import Path

visa_root = Path("D:/ECR-AD/datasets/visa/VisA/data/VisA_20220922")
categories = ["candle", "capsules", "cashew", "chewinggum", "fryum",
              "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum"]

for cat in categories:
    normal_dir = visa_root / cat / "Data" / "Images" / "Normal"
    anomaly_dir = visa_root / cat / "Data" / "Images" / "Anomaly"
    mask_dir = visa_root / cat / "Data" / "Masks" / "Anomaly"
    
    n_normal = len(list(normal_dir.glob("*.JPG"))) + len(list(normal_dir.glob("*.jpg")))
    n_anomaly = len(list(anomaly_dir.glob("*.JPG"))) + len(list(anomaly_dir.glob("*.jpg")))
    
    print(f"{cat:15s} | Normal: {n_normal:4d} | Anomaly: {n_anomaly:4d}")