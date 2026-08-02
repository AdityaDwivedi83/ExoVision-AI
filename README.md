# 🌌 ExoVision-AI

> End-to-end AI pipeline for detecting exoplanet candidates from raw NASA telescope observations.

![Python](https://img.shields.io/badge/Python-3.9-blue)
![PyTorch](https://img.shields.io/badge/PyTorch-2.x-red)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green)
![Status](https://img.shields.io/badge/Status-In%20Development-orange)

---

## 🚀 Overview

ExoVision-AI is an end-to-end machine learning project that aims to automate the process of detecting exoplanet candidates from raw telescope observations.

Instead of training directly on preprocessed light-curve datasets, this project reconstructs the scientific pipeline used in modern astronomy by converting raw telescope image sequences into light curves before applying deep learning models for exoplanet classification.

The goal is to bridge computer vision, astronomical image processing, photometry, and time-series machine learning into a single reproducible workflow.

---

## 🔭 Pipeline

```text
NASA MAST Archive
        │
        ▼
Download Target Pixel Files (.fits)
        │
        ▼
Read FITS Image Cubes
        │
        ▼
Computer Vision
Locate Target Star
        │
        ▼
Aperture Photometry
Measure Brightness
        │
        ▼
Generate Light Curves
        │
        ▼
Deep Learning
(1D CNN / LSTM / Transformer)
        │
        ▼
Predict Exoplanet Probability
```

---

## 🛰 Features

- Read NASA FITS files
- Download Kepler/TESS observations from MAST
- Target star localization
- Aperture photometry
- Automatic light curve generation
- Deep learning-based exoplanet classification
- Interactive visualization
- Web deployment

---

## 🛠 Tech Stack

- Python
- NumPy
- Pandas
- OpenCV
- Astropy
- Lightkurve
- Photutils
- PyTorch
- Matplotlib
- Streamlit

---

## 📁 Project Structure

```
ExoVision-AI/

├── app/
├── data/
├── docs/
├── models/
├── notebooks/
├── src/
├── tests/
├── README.md
└── requirements.txt
```

---

## 🚧 Current Progress

- [x] Project setup
- [x] Virtual environment
- [x] Repository structure
- [x] Dependency installation
- [ ] Download data from MAST
- [ ] Read FITS files
- [ ] Target star detection
- [ ] Aperture photometry
- [ ] Light curve generation
- [ ] Model training
- [ ] Deployment

---

## First Kepler Observation

The figure below shows the first raw observation frame downloaded from NASA's Kepler mission. The red marker indicates the automatically detected brightest pixel corresponding to the target star.

![Kepler Frame](docs/images/kepler_first_frame.png)

## 🎯 Long-Term Goal

Build a complete AI system capable of converting raw astronomical observations into exoplanet probability predictions while maintaining a transparent and reproducible scientific workflow.

---

## 📜 License

MIT License