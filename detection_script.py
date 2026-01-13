import os
import cv2
import json
import argparse
from ultralytics import YOLO

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input", type=str, required=True,
        help="Input images folder"
    )
    parser.add_argument(
        "--output", type=str, default="output",
        help="Output folder for annotated images"
    )
    parser.add_argument(
        "--confidence", type=float, default=0.5,
        help="Confidence threshold"
    )
    return parser.parse_args()

def process_images():
    args = parse_args()

    # ✅ Create output and logs directories if they don't exist
    os.makedirs(args.output, exist_ok=True)
    logs_dir = os.path.join("logs")
    os.makedirs(logs_dir, exist_ok=True)

    # ✅ Load YOLOv8 pretrained model (will auto-download if needed)
    model = YOLO("yolov8n.pt")

    # ✅ Class mapping
    class_map = {0: "gloved_hand", 1: "bare_hand"}

    # ✅ Loop through all images in input folder
    for img_name in os.listdir(args.input):
        if not (img_name.lower().endswith(".jpg") or img_name.lower().endswith(".png")):
            continue

        img_path = os.path.join(args.input, img_name)
        image = cv2.imread(img_path)

        if image is None:
            print(f"[!] Failed to read {img_name}")
            continue

        # ✅ Run detection
        results = model(image, conf=args.confidence)[0]
        detections = []

        for box in results.boxes:
            cls_id = int(box.cls[0])
            label = class_map.get(cls_id, "unknown")
            conf = float(box.conf[0])
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())

            detections.append({
                "label": label,
                "confidence": round(conf, 2),
                "bbox": [x1, y1, x2, y2]
            })

            # ✅ Draw bounding box
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(
                image,
                f"{label} {conf:.2f}",
                (x1, y1 - 10),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 255, 0),
                2
            )

        # ✅ Save annotated image
        cv2.imwrite(os.path.join(args.output, img_name), image)

        # ✅ Save JSON log in logs folder
        log_data = {"filename": img_name, "detections": detections}
        json_path = os.path.join(logs_dir, img_name.rsplit(".", 1)[0] + ".json")
        with open(json_path, "w") as f:
            json.dump(log_data, f, indent=2)

        print(f"[✓] Processed {img_name} - {len(detections)} detections")

if __name__ == "__main__":
    process_images()
