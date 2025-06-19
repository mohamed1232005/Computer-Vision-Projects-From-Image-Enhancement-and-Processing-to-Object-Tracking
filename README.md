# Computer-Vision-Projects-From-Image-Enhancement-and-Processing-to-Object-Tracking

## Computer Vision Projects – Implemented with Python, OpenCV, NumPy, and YOLO

A comprehensive set of five professional projects developed as part of a **Computer Vision** curriculum, covering essential domains from **image enhancement** to **object tracking**. Each project focuses on specific processing techniques and real-world applications using Python-based tools and libraries such as **OpenCV**, **NumPy**, **Matplotlib**, **cuML**, **Ultralytics YOLO**, and more.

---

## 📁 Contents

Each directory contains:
- Full Python notebooks
- Output images or videos
- Detailed implementations with experiments
- (Optional) Project reports

| Folder Name                                                           | Technique Name                                                              | Domain             | Category            |
|------------------------------------------------------------------------|------------------------------------------------------------------------------|---------------------|---------------------|
| `spatial-enhancement-brightness-contrast`                             | Spatial Image Enhancement with Brightness and Contrast Adjustments          | Image Enhancement   | Spatial Domain      |
| `frequency-filtering-fourier-transform`                               | Frequency Domain Filtering Using Discrete Fourier Transform                 | Image Enhancement   | Frequency Domain    |
| `morphological-image-processing`                                      | Morphological Image Processing: Erosion, Dilation, Opening, and Closing     | Image Processing    | Shape Analysis      |
| `image-retrieval-sift-kmeans-tfidf`                                   | Content-Based Image Retrieval Using SIFT, K-Means, and TF-IDF               | Image Retrieval     | Feature Matching    |
| `video-object-tracking-yolov10n-v11n-v12n`                            | Real-Time Object Tracking in Video Using YOLOv10n, YOLOv11n, and YOLOv12n   | Video Processing    | Object Detection    |

---

## 🧠 Summary of Techniques

| Technique                                          | Domain             | Method Type         | Tools & Libraries Used                                    |
|---------------------------------------------------|--------------------|----------------------|-----------------------------------------------------------|
| Brightness & Contrast Enhancement                 | Image Enhancement  | Spatial              | OpenCV, NumPy, Matplotlib                                 |
| Fourier-Based Filtering                           | Image Enhancement  | Frequency            | OpenCV, NumPy, DFT, FFT                                   |
| Morphological Operations                          | Shape Processing   | Structural           | OpenCV, Custom Kernel Logic                               |
| Feature-Based Image Retrieval                     | Retrieval/Search   | SIFT + Clustering    | OpenCV SIFT, cuML KMeans, Cosine Similarity, TF-IDF       |
| YOLO-Based Object Tracking in Video               | Real-Time Vision   | Deep Learning        | Ultralytics YOLO, OpenCV, FFmpeg, Google Colab            |

---

## 📦 `spatial-enhancement-brightness-contrast`

### 📌 Description
Performs **brightness** and **contrast** enhancement using both NumPy and OpenCV's `cv2.add()` methods with proper handling of data types (`uint8`) and overflow.

### ⚙️ Implemented Features
- Linear brightness control via offset
- Contrast control via scaling factor
- Combined contrast+brightness enhancement
- Histogram Equalization (grayscale + color)
- CLAHE (Contrast Limited Adaptive Histogram Equalization)
- Median, Gaussian, and Sharpen filters
- Adaptive vs Global Thresholding comparison

### 📁 Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib

---

## 📦 `frequency-filtering-fourier-transform`

### 📌 Description
Applies **Fourier Transform** to shift images into the frequency domain, allowing for:
- Low-pass filtering (blurring)
- High-pass filtering (edge enhancement)
- Band-pass and band-stop filtering

### 🔍 Methods
- `cv2.dft()` + `np.fft.fftshift()`
- Custom filters in frequency space
- Inverse DFT to reconstruct spatial domain

### 📁 Technologies Used
- OpenCV
- NumPy
- Matplotlib

---

## 📦 `morphological-image-processing`

### 📌 Description
Binary and grayscale morphological operations:
- Erosion
- Dilation
- Opening
- Closing

Implemented using both **OpenCV** and **custom functions** for educational insight.

### ⚙️ Kernel
Used a 3×3 structuring element. Explored effects on multiple input textures.

### 📁 Technologies Used
- Python
- OpenCV
- NumPy
- Matplotlib

---

## 📦 `image-retrieval-sift-kmeans-tfidf`

### 📌 Description
Performs **content-based image retrieval** using:
- **SIFT** descriptors
- **KMeans clustering (cuML GPU)** to form visual words
- **BoW vs. TF-IDF encoding**
- **Cosine Similarity** for ranking

### 🧪 Experiments
| Experiment                        | Variable                  | Result                             |
|----------------------------------|---------------------------|------------------------------------|
| Image Count for Centroids        | 500 / 1000 / 2000         | 1000 chosen as best trade-off      |
| KMeans Centroids (Visual Words)  | 50 / 150 / 250            | 150 found most accurate            |
| Encoding Comparison              | BoW vs. TF-IDF            | TF-IDF outperformed in accuracy    |

### 📁 Technologies Used
- OpenCV SIFT
- cuML, CuPy
- Scikit-learn
- Matplotlib

---

## 📦 `video-object-tracking-yolov10n-v11n-v12n`

### 📌 Description
Compares three YOLO variants on video input:
- `YOLOv10n.pt`
- `YOLOv11n.pt`
- `YOLOv12n.pt`

Runs frame-by-frame tracking, measures total inference time, FPS, and output quality.

### 🧪 Results

| Model     | Total Time (s) | Frames | FPS  | Notes                 |
|-----------|----------------|--------|------|------------------------|
| YOLOv10n  | 203.51         | 942    | 4.63 | Moderate speed         |
| YOLOv11n  | 190.74         | 942    | 4.94 | Fastest overall        |
| YOLOv12n  | 210.47         | 942    | 4.48 | Heaviest but accurate  |

### 🧰 Tools & Formats
- Ultralytics YOLO
- Google Colab
- OpenCV
- FFmpeg (convert `.avi` → `.mp4`)

---

## 🚀 Getting Started

Install dependencies:
```bash
pip install opencv-python matplotlib numpy cupy cuml scikit-learn ultralytics
```

---
