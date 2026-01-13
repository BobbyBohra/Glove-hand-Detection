# Gloved vs Bare Hand Detection

## Dataset
Public hand and glove images sourced from Roboflow Universe and internet images.

## Model
YOLOv8 (pretrained yolov8n)

## Approach
A pretrained YOLOv8 model is used for object detection. The model detects hands and classifies them as gloved or bare hands based on visual features.

## Preprocessing
- Images resized internally by YOLO
- Confidence threshold filtering

## Output
- Annotated images saved in `output/`
- Detection logs saved as JSON files in `logs/`

## What Worked
- YOLOv8 provided fast and accurate inference
- CLI-based script allows easy reuse

## Limitations
- Model is pretrained and not fine-tuned specifically on glove datasets
- Edge cases like occluded hands may fail

## How to Run
```bash
python detection_script.py --input images --output output --confidence 0.5




---


