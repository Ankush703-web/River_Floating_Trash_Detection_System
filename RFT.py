import cv2
import torch
from detectron2.engine import DefaultPredictor
from detectron2.config import get_cfg
from detectron2 import model_zoo
from detectron2.utils.visualizer import Visualizer
from detectron2.data import MetadataCatalog

# === Load config and model ===
cfg = get_cfg()
cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_R_50_FPN_3x.yaml"))

# Use your local trained model checkpointR
cfg.MODEL.WEIGHTS = "RFT_model.pth"  # path to trained model
cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5  # threshold for predictions
cfg.MODEL.ROI_HEADS.NUM_CLASSES = 8  # number of classes in your dataset
cfg.DATASETS.TEST = ("my_dataset_test", )
MetadataCatalog.get("my_dataset_test").thing_classes = [
    "bottle", "grass", "branch", "milk-box", "plastic-bag",
    "plastic-garbage", "ball", "leaf"
]

predictor = DefaultPredictor(cfg)
metadata = MetadataCatalog.get(cfg.DATASETS.TEST[0])

# === Open the video ===
video_path = "RFT1.mp4"   
cap = cv2.VideoCapture(video_path)

# === Define output video writer ===
output_path = "RFT1_output.mp4"
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
fps = cap.get(cv2.CAP_PROP_FPS)
width  = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

# === Process each frame ===
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    outputs = predictor(frame)
    v = Visualizer(frame[:, :, ::-1], metadata=metadata, scale=1.0)
    result = v.draw_instance_predictions(outputs["instances"].to("cpu"))

    out.write(result.get_image()[:, :, ::-1])  # Save the frame to output

# === Release everything ===
cap.release()
out.release()
cv2.destroyAllWindows()
print(f"Inference completed. Output saved to {output_path}")
