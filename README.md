# Audio Classification with Hardware-Efficient RNNs 🎤🤖

[![Python](https://img.shields.io/badge/python-3.10-blue?logo=python)](https://www.python.org/)
[![TensorFlow](https://img.shields.io/badge/tensorflow-2.14-orange?logo=tensorflow)](https://www.tensorflow.org/)

---

## Project Overview

This project implements a **hardware-efficient Vanilla Recurrent Neural Network (RNN)** for audio classification, specifically targeting the Google Speech Commands dataset. The focus is on:

- Lightweight Vanilla RNN architecture (no LSTMs or GRUs)
- 8-bit post-training weight quantization for edge deployment
- Gradient stabilization techniques (Adam optimizer, gradient clipping, L2 regularization, Dropout)

**Goal:** Classify 10 core speech commands (`yes, no, up, down, left, right, on, off, stop, go`) with high accuracy while minimizing model size.

---

## Key Results

| Model                 | Accuracy | Model Size | Notes                        |
|-----------------------|---------|------------|-------------------------------|
| Full Precision RNN    | XX%     | 326 KB     | Baseline                     |
| Quantized RNN (8-bit) | YY%     | 50 KB      | Optimized for hardware       |

---

## Project Structure

| File                           | Description                                           |
|--------------------------------|------------------------------------------------------|
| `data_preprocessing1.py`       | Dataset download, spectrogram generation, preprocessing |
| `rnn_model1.py`                | Vanilla RNN model definition, training, evaluation, quantization |
| `full_precision_rnn_model.h5`  | Trained full-precision model                         |
| `quantized_rnn_model.tflite`   | 8-bit quantized model for edge deployment           |
| `vanilla_rnn_report.pdf`       | PDF report with performance metrics and analysis    |
| `README.md`                     | Project overview and instructions                   |
| `requirements.txt`             | Python dependencies                                  |

---

## How to Run

1. **Clone the repository**
   ```bash
   git clone https://github.com/GausicVC/hardware-efficient-rnn-speech-command-detection.git
   cd hardware-efficient-rnn-speech-command-detection
