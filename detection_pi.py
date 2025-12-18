# -*- coding: utf-8 -*-
import time
import sys
import cv2
import numpy as np
import tflite_runtime.interpreter as tflite
from picamera2 import Picamera2

# Load COCO labelmap
LABELMAP_PATH = "labelmap.txt"
with open(LABELMAP_PATH, "r") as f:
    LABELMAP = [line.strip() for line in f.readlines()]

def detect_from_camera():
    print("1. Loading Model...")
    interpreter = tflite.Interpreter(model_path="detect.tflite")
    interpreter.allocate_tensors()
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    print("2. Model Loaded!")

    print("3. Starting Camera (Picamera2 Native Mode)...")
    
    try:
        picam2 = Picamera2()
        # Using Video Configuration
        config = picam2.create_video_configuration(main={"size": (640, 480), "format": "RGB888"})
        picam2.configure(config)
        picam2.start()
        print("4. Camera Active! Press 'q' to quit.")
    except Exception as e:
        print(f"ERROR Starting Camera: {e}")
        return

    while True:
        try:
            # 1. Capture Image
            img_rgb = picam2.capture_array()
            
            # 2. Prepare for AI (Resize)
            img_resized = cv2.resize(img_rgb, (300, 300))
            input_data = np.expand_dims(img_resized, axis=0)

            # 3. Run Inference
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()

            # 4. Get Results
            boxes  = interpreter.get_tensor(output_details[0]['index'])
            labels = interpreter.get_tensor(output_details[1]['index'])
            scores = interpreter.get_tensor(output_details[2]['index'])
            nums   = interpreter.get_tensor(output_details[3]['index'])

            # 5. Prepare Image for Display (Convert RGB to BGR for OpenCV)
            img_display = cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR)

            found_something = False
            
            # 6. Draw Boxes on the Screen
            for i in range(int(nums[0])):
                if scores[0][i] > 0.5:
                    class_id = int(labels[0][i])
                    label_name = LABELMAP[class_id] if 0 <= class_id < len(LABELMAP) else "unknown"
                    
                    # Math to draw the box
                    ymin, xmin, ymax, xmax = boxes[0][i]
                    h, w, _ = img_display.shape
                    x0, y0 = int(xmin * w), int(ymin * h)
                    x1, y1 = int(xmax * w), int(ymax * h)

                    # Draw Rectangle and Text
                    cv2.rectangle(img_display, (x0, y0), (x1, y1), (0, 255, 0), 2)
                    cv2.putText(img_display, f"{label_name} {scores[0][i]*100:.0f}%", (x0, y0-10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

                    print(f"I see: {label_name}")
                    found_something = True
            
            # 7. SHOW THE WINDOW
            cv2.imshow('Drone Vision', img_display)

            # Press 'q' to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        except KeyboardInterrupt:
            break
        except Exception as e:
            print(f"Loop Error: {e}")
            break

    picam2.stop()
    cv2.destroyAllWindows()
    print("\nStopped.")

if __name__ == "__main__":
    detect_from_camera()