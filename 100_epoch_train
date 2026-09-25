# 1. Install the ultralytics library
!pip install -q ultralytics

# 2. Import YOLO and start training
from ultralytics import YOLO

# Load pre-trained YOLO11 Nano model
model = YOLO('yolo11n.pt')

# Train on your dataset
results = model.train(
    data='Underwater-Crack-Detection-10/data.yaml',
        epochs=100,  # Increased from 50
            imgsz=512,
                batch=16,
                    device=0,
                        patience=20  # Added early stopping
                        )
