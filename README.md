# 🛠️ HAL Parts Defect Detection System

An AI-powered web application for **automated defect detection in aircraft parts**, developed for **Hindustan Aeronautics Limited (HAL)**.  
The system combines **deep learning (YOLOv8)**, **SSIM-based image comparison**, and **deep image alignment (SuperPoint, SIFT, ORB)** to identify surface defects with aerospace-grade precision.

---

## ✨ Key Highlights

🔹 Detect defects using **both image comparison & deep learning**  
🔹 Upload **multiple reference & multiple test images** for multi-angle inspection  
🔹 Optional upload of **custom YOLOv8 model (.pt)** for specialized defect categories  
🔹 **Visual and tabular reports** for every (reference, test) pair  
🔹 **Side-by-side comparison** of annotated output for SSIM & YOLO  
🔹 **Downloadable CSV** (pair-wise and summary)  
🔹 **Configurable detection sensitivity** from the sidebar  
🔹 **Government-grade UI**, disclaimers & secure workflow  

---

## 🧠 Core Features

| Feature | Description |
|--------|-------------|
| SSIM Defect Detection | Pixel-level similarity comparison with defect heatmaps |
| YOLOv8 Deep Learning | Bounding box detection, labels & confidence scores |
| Deep Alignment | Corrects rotation/zoom using SuperPoint → SIFT → ORB → Template |
| Color Detection | LAB / DeltaE scoring for subtle surface tone changes |
| Pattern Matching | ORB pattern difference detection |
| Reporting | Download CSV (per-pair & summary) |

---

## 📂 Project Structure (Conceptual)

```

HAL-Defect-Detection/
│
├── M.py                  # Classic UI
├── Main.py               # Enhanced UI + better logging & UX
├── hal_logo.png
├── requirement.txt
└── (first-run auto) superpoint_v1.pth

````

> **Both `M.py` and `Main.py` provide the complete workflow.**  
> `Main.py` offers richer interaction, feedback & logging; `M.py` is a streamlined interface.

---

## 📦 Requirements

See `requirement.txt` for the full dependency list.

Additional notes:
- `torch` is required for **SuperPoint** deep alignment (auto-downloads weights on first run)
- Recommended Python version: **3.8 – 3.11**
- Python **3.12+ may fail** due to unbuilt wheels for OpenCV/Torch

---

## ⚙️ Installation Guide

```bash
# 1. Create virtual environment
python -m venv venv

# Activate
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate

# 2. Install dependencies
pip install -r requirement.txt
````

Verify installations:

```bash
python --version
pip show streamlit opencv-python torch ultralytics scikit-image
```

---

## ▶️ Run the Application

### Option 1 – Classic UI

```bash
streamlit run M.py
```

### Option 2 – Enhanced UI (recommended)

```bash
streamlit run Main.py
```

---

## 📝 Usage Instructions

1. Upload **one or more reference images** (defect-free component)
2. Upload **one or more test images** (to be inspected)
3. (Optional) Upload **custom YOLOv8 `.pt` model**
4. Adjust **SSIM sensitivity, color threshold, pattern threshold & YOLO confidence**
5. Toggle detection modules:

   * 🔲 Color / DeltaE
   * 🔲 Pattern / ORB
   * 🔲 YOLOv8 deep detection
6. Review:

   * **SSIM heatmap & bounding contours**
   * **YOLO detected bounding boxes**
   * **Summary & detailed tables**
7. Download **CSV defect reports**

---

## 🧠 Approach & Algorithms

| Step            | Algorithms                                    | Libraries            |
| --------------- | --------------------------------------------- | -------------------- |
| Alignment       | SuperPoint → SIFT → ORB → Template Matching   | Torch, OpenCV        |
| Similarity      | SSIM + adaptive threshold + contour detection | Scikit-image, OpenCV |
| Deep Learning   | YOLOv8                                        | Ultrayltics          |
| Color Defects   | ΔE LAB scoring                                | OpenCV               |
| Pattern Defects | ORB                                           | OpenCV               |
| Reporting       | DataFrame + CSV                               | pandas               |

The system warns the user if alignment confidence is low and logs difficult cases for operator review.

---

## 🔐 Disclaimer

> This is an **official government-grade application**.
> Unauthorized access or misuse is strictly prohibited and may be punishable under applicable law.
> Images uploaded are processed **only for defect detection** and are **not stored permanently**.

---

## 🏛️ About HAL

**Hindustan Aeronautics Limited (HAL)** is an Indian state-owned aerospace and defence corporation engaged in the design, development, and manufacture of aircraft, jet engines, helicopters, and related components.

---

## 👨‍💻 Developers

This project was collaboratively developed by:

* **Ishaan Tripathi**
* **Abhiyanshu Anand**


---

## ⭐ If you find this project useful

Please consider giving the repository a **star** on GitHub — it helps support development and visibility of public research projects.

