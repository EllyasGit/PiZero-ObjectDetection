# -*- coding: utf-8 -*-
import cv2
import tensorflow as tf
import numpy as np

# Load COCO labelmap (80 classes)
LABELMAP_PATH = "labelmap.txt"
with open(LABELMAP_PATH, "r") as f:
    LABELMAP = [line.strip() for line in f.readlines()]

def detect_from_camera():
    interpreter = tf.lite.Interpreter(model_path="detect.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    cap = cv2.VideoCapture(0)

    while True:
        ret, img_org = cap.read()
        if not ret:
            continue

        key = cv2.waitKey(1)
        if key == 27:  # ESC
            break

        # Preprocess input
        img = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (300, 300))
        img = img.reshape(1, 300, 300, 3).astype(np.uint8)

        interpreter.set_tensor(input_details[0]['index'], img)

        interpreter.invoke()

        boxes  = interpreter.get_tensor(output_details[0]['index'])
        labels = interpreter.get_tensor(output_details[1]['index'])
        scores = interpreter.get_tensor(output_details[2]['index'])
        nums   = interpreter.get_tensor(output_details[3]['index'])

        for i in range(int(nums[0])):
            if scores[0][i] > 0.5:
                ymin, xmin, ymax, xmax = boxes[0][i]
                x0 = int(xmin * img_org.shape[1])
                y0 = int(ymin * img_org.shape[0])
                x1 = int(xmax * img_org.shape[1])
                y1 = int(ymax * img_org.shape[0])

                class_id = int(labels[0][i])

                # FIX: map class ID to label
                if 0 <= class_id < len(LABELMAP):
                    label_name = LABELMAP[class_id]
                else:
                    label_name = "unknown"

                cv2.rectangle(img_org, (x0, y0), (x1, y1), (255, 0, 0), 2)

                text = f"{label_name} {scores[0][i]*100:.1f}%"
                cv2.putText(img_org, text, (x0, max(15, y0-10)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        cv2.imshow("SSD MobileNet Detection", img_org)

    cap.release()
    cv2.destroyAllWindows()


if __name__ == '__main__':
    detect_from_camera()
