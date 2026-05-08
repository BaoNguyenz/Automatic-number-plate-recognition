# 🚗 Automatic Number Plate Recognition (ANPR)

A robust Computer Vision pipeline for real-time vehicle tracking and automatic license plate recognition using **YOLOv8**, **SORT**, and **EasyOCR**. 

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)
![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-blue)
![OpenCV](https://img.shields.io/badge/OpenCV-Image_Processing-green)
![EasyOCR](https://img.shields.io/badge/EasyOCR-Text_Extraction-orange)
![License](https://img.shields.io/badge/License-MIT-yellow)

---

## ✨ Features

- 🚙 **Vehicle Detection:** Utilizes a pre-trained YOLOv8 model to accurately detect vehicles (cars, buses, trucks, etc.) in video frames.
- 🎯 **License Plate Detection:** Employs a custom-trained YOLOv8 model specifically fine-tuned to locate license plates.
- 🔄 **Real-Time Tracking:** Integrates the **SORT (Simple Online and Realtime Tracking)** algorithm to track vehicles across consecutive frames, assigning unique IDs to each.
- 🔤 **Optical Character Recognition (OCR):** Uses **EasyOCR** to extract the alphanumeric text from the cropped license plate regions.
- 🛠️ **Image Processing Pipeline:** Applies grayscale conversion and binary thresholding via OpenCV to enhance image quality before OCR processing.
- 📊 **CSV Export:** Outputs tracking IDs, bounding boxes, and recognized license plate text (with confidence scores) to a structured CSV file.

---

## 🏗️ System Pipeline

1. **Input:** Read frames from a video source.
2. **Vehicle Detection:** YOLOv8 identifies and draws bounding boxes around vehicles.
3. **Tracking:** SORT algorithm assigns and maintains a unique ID for each detected vehicle across frames.
4. **License Plate Detection:** A secondary YOLOv8 model finds the license plate within the vehicle's bounding box.
5. **Preprocessing:** The license plate image is cropped, converted to grayscale, and thresholded for better text clarity.
6. **OCR:** EasyOCR reads the text. A custom formatting function ensures the text complies with standard license plate formats (correcting common OCR mistakes like confusing 'O' with '0').
7. **Output:** Results are saved frame-by-frame into a CSV file.

---

## 🛠️ Tech Stack

| Component | Technology |
|-----------|------------|
| **Object Detection** | YOLOv8 (Ultralytics) |
| **Object Tracking** | SORT |
| **OCR** | EasyOCR |
| **Image Processing** | OpenCV |
| **Data Handling** | NumPy, Pandas |

---

## 📁 Project Structure

```
Automatic number plate recognition/
├── main.py                  # Main execution script
├── util.py                  # Utility functions (OCR, formatting, CSV writing)
├── visualize.py             # Script to visualize results on the video
├── add_missing_data.py      # Script to interpolate missing tracking frames
├── environment.yml          # Conda environment configuration
├── sort/                    # SORT tracking algorithm module
├── Licence_plate_detection/ # YOLOv8 models (pretrained and custom weights)
├── Csv_results/             # Output directory for CSV files
└── README.md                # Project documentation
```

---

## 🚀 Installation & Setup

### 1. Clone the Repository

```bash
git clone <repository-url>
cd "Automatic number plate recognition"
```

### 2. Set Up Environment

It is recommended to use Conda:

```bash
conda env create -f environment.yml
conda activate <env-name>
```

*(Alternatively, you can install the main packages via pip: `pip install ultralytics opencv-python easyocr filterpy sort`)*

### 3. Run the Pipeline

1. Place your target video in the root directory (e.g., `2103099-uhd_3840_2160_30fps.mp4`).
2. Execute the main script to process the video and generate the CSV output:

```bash
python main.py
```

### 4. Process Data & Visualize

After generating `test.csv`, you can optionally interpolate missing data and visualize the tracking boxes and OCR text directly onto the video using the provided utility scripts (`add_missing_data.py` and `visualize.py`).

---

## 📄 License

This project is licensed under the **MIT License**.