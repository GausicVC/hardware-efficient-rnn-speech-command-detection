# Audio Classification with Hardware-Efficient RNNs 🎤🤖

## Project Overview

This project implements a **hardware-efficient Vanilla Recurrent Neural Network (RNN)** for audio classification, specifically targeting the Google Speech Commands dataset. The focus is on:

- Lightweight Vanilla RNN architecture (no LSTMs or GRUs)
- 8-bit post-training weight quantization for edge deployment
- Gradient stabilization techniques (Adam optimizer, gradient clipping, L2 regularization, Dropout)

**Goal:** Classify 10 core speech commands (`yes, no, up, down, left, right, on, off, stop, go`) with high accuracy while minimizing model size.

---

## Project Structure

| File                           | Description                                           |
|--------------------------------|------------------------------------------------------|
| `data_preprocessing1.py`       | Dataset download, spectrogram generation, preprocessing |
| `rnn_model1.py`                | Vanilla RNN model definition, training, evaluation, quantization |
| `full_precision_rnn_model.h5`  | Trained full-precision model                         |
| `quantized_rnn_model.tflite`   | 8-bit quantized model for edge deployment           |
| `vanilla_rnn_report.pdf`       | PDF report with performance metrics and analysis    |
| `Complete_Guide.md`            | Project overview and instructions                   |


---

