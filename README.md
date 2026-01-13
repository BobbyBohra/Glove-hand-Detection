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

# 🧠 PART 2: Reasoning Answers (Part_2_Answers.md)

```markdown
## Q1: Choosing the Right Approach
I would use object detection because the task requires locating whether a label exists on the product, not just classifying the entire image. Detection allows us to check both presence and position of the label. Classification may fail if the label is small or partially visible. Segmentation would be overkill unless pixel-level accuracy is required. If detection fails due to lighting or occlusion, segmentation could be used as a fallback.

## Q2: Debugging a Poorly Performing Model
I would first compare training and test images to check for distribution differences such as lighting or camera angle. Next, I would visualize predictions and confidence scores to identify systematic errors. Checking class imbalance and annotation quality is important. I would also evaluate overfitting using training vs validation metrics. Finally, I would test the model on a small curated validation set.

## Q3: Accuracy vs Real Risk
Accuracy is not the right metric because missing defective products has a high real-world cost. Recall is more important, especially for defective or unsafe cases. A high false negative rate can be dangerous in production systems. Precision-recall tradeoffs and confusion matrix should be analyzed. Metrics like Recall, F1-score, and False Negative Rate are more meaningful here.

## Q4: Annotation Edge Cases
Blurry or partially visible objects should be included if they reflect real-world conditions. Excluding them may result in a model that performs well only on clean data. However, too many poor-quality samples can introduce noise. A balanced approach is to include them but label them carefully. This improves robustness while maintaining data quality.
