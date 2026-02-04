
# 🚁 TensorFlow Lite Object Detection (Raspberry Pi Zero 2 W & PC)

This project implements real-time object detection using **TensorFlow Lite** and **MobileNet SSD**. It is designed to be cross-platform, with specific optimizations to run smoothly on the low-memory **Raspberry Pi Zero 2 W**, as well as standard PCs/Laptops.

Whether you are building an AI drone, a security camera, or just learning computer vision, this repo provides the "Native" implementation for maximum performance.

## 📂 Project Structure

  * `detection_pi.py` → Optimized code for Raspberry Pi (Uses `Picamera2` & `libcamera`).
  * `detection_pc.py` → Standard code for PC/Laptop (Uses `OpenCV`).
  * `requirements_pi.txt` → Dependencies for Raspberry Pi.
  * `requirements_pc.txt` → Dependencies for PC.
  * `detect.tflite` → The Quantized MobileNet SSD model (Lightweight AI).
  * `labelmap.txt` → List of objects the model can detect (COCO dataset).

-----

## 🍓 Guide for Raspberry Pi Zero 2 W

Running AI on a Pi Zero 2 W is challenging due to the 512MB RAM limit. This guide uses the **Native Picamera2** library (bypassing OpenCV for video capture) to save memory and prevent crashes.

### 1\. System Configuration (CRITICAL)

Before running the code, you **must** configure your Pi to prevent "Out of Memory" kills.

  * **Increase Swap to 2GB:**

    1.  `sudo nano /etc/dphys-swapfile`
    2.  Change `CONF_SWAPSIZE=100` to `CONF_SWAPSIZE=2048`
    3.  Restart swap: `sudo /etc/init.d/dphys-swapfile restart`

  * **Disable Legacy Camera:**

    1.  `sudo nano /boot/firmware/config.txt`
    2.  **Remove/Comment out:** `start_x=1` and `gpu_mem=128`.
    3.  **Ensure enabled:** `camera_auto_detect=1`.
    4.  Reboot: `sudo reboot`

### 2\. Installation

We use the system's pre-installed libraries to avoid compiling heavy packages.

1.  **Install System Dependencies:**

    ```bash
    sudo apt update
    sudo apt install -y python3-opencv python3-picamera2 libcamera-tools
    ```

2.  **Create the Virtual Environment:**
    We use the `--system-site-packages` flag to let Python see the installed camera tools.

    ```bash
    python3 -m venv --system-site-packages obj_detect_env
    source obj_detect_env/bin/activate
    ```

3.  **Install Python Libraries:**

    ```bash
    pip install -r requirements_pi.txt
    ```

### 3\. Running the Code

```bash
source obj_detect_env/bin/activate
python detection_pi.py
```

**Note:** If running via SSH without a monitor, the video window might fail. To run in "Headless Mode" (faster FPS), edit `detection_pi.py` and comment out the `cv2.imshow` lines.

-----

## 💻 Guide for PC (Windows / Mac / Linux)

Running on a PC is straightforward and uses standard USB webcam input.

### 1\. Installation

1.  **Create a virtual environment (Optional but recommended):**

    ```bash
    python -m venv venv
    # Windows: venv\Scripts\activate
    # Mac/Linux: source venv/bin/activate
    ```

2.  **Install Dependencies:**

    ```bash
    pip install -r requirements_pc.txt
    ```

### 2\. Running the Code

Ensure your webcam is plugged in.

```bash
python detection_pc.py
```

-----

## ⚡ Performance Notes

  * **PC:** Should run at 30+ FPS depending on your CPU/GPU.
  * **Pi Zero 2 W:**
      * **With GUI Window:** \~1-2 FPS (Laggy due to drawing graphics).
      * **Headless (No Window):** \~4-6 FPS (Good for automated drones/bots).
      * **With Google Coral USB:** \~20-30 FPS (Requires Edge TPU hardware).

## 🛠 Troubleshooting

  * **"Killed" error on Pi:** Your Swap memory is likely not active, or you are trying to use the PC code on the Pi.
  * **"Libcamera not found":** Make sure you are using the `--system-site-packages` flag when creating the environment.
  * **Camera not opening:** Run `libcamera-hello` in the terminal to verify your hardware connection.

-----

### Author

Created by EllyasGit for the "Object Detection" Project.