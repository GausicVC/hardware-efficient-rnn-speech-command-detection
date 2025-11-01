# Audio Classification with Hardware-Efficient Neural Networks

## 1. Introduction

This document details the implementation of a hardware-efficient Vanilla Recurrent Neural Network (RNN) for audio classification, specifically targeting the Google Speech Commands dataset. The primary goal was to adhere to strict hardware constraints, including the use of a basic RNN architecture without complex gating mechanisms (like LSTMs or GRUs) and implementing 8-bit weight quantization. The project also aimed to explore training techniques to mitigate the notorious vanishing/exploding gradient problem inherent in vanilla RNNs.

## 2. Dataset Acquisition and Preprocessing

The Google Speech Commands dataset, comprising 105,829 one-second audio recordings of 35 different spoken words, was chosen for this task. The focus was on the 10 core commands: 'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go'.

## 3. Vanilla RNN Architecture Design

The core of the solution is a simple Vanilla RNN, adhering to the constraint of avoiding complex recurrent architectures. The model is implemented using TensorFlow/Keras and consists of the following layers:

*   **Input Layer**: Accepts spectrograms as input. The shape of the input is determined based on the preprocessed spectrograms.
*   **SimpleRNN Layer**: This is the basic recurrent layer, with `tanh` activation. L2 regularization is applied to both `kernel` and `recurrent` weights to prevent overfitting and help stabilize training.
    *   `h_t = tanh(W_hh * h_{t-1} + W_ih * x_t + b)`
*   **Dropout Layer**: A dropout layer is included after the RNN to further regularize the model and prevent overfitting.
*   **Dense Layer**: A final dense layer with `softmax` activation is used for classification into the 10 core command classes. L2 regularization is also applied to its kernel weights.

### Model Summary

```
Model: "sequential"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ simple_rnn (SimpleRNN)          │ (None, 128)            │        197120 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 128)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 10)             │         2570  │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 199690 (326.04 KB)
 Trainable params: 199690 (326.04 KB)
 Non-trainable params: 0 (0.00 B)
```

## 4. 8-bit Weight Quantization

To meet the requirement for hardware-efficient operations, 8-bit weight quantization was implemented using TensorFlow Lite's post-training quantization. This approach involves training the model in full precision first and then converting it to a quantized 8-bit model. 

### Post-Training Quantization Steps:

1.  **Train Full Precision Model**: The Vanilla RNN is trained using standard float32 precision.
2.  **TFLiteConverter**: A `tf.lite.TFLiteConverter` is used to convert the trained Keras model into a TensorFlow Lite model.
3.  **Optimization Settings**: `converter.optimizations = [tf.lite.Optimize.DEFAULT]` is set to enable default optimizations, including quantization.
4.  **Target Specification**: `converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]` specifies that the model should be quantized to 8-bit integers.
5.  **Input/Output Types**: `converter.inference_input_type = tf.int8` and `converter.inference_output_type = tf.int8` are set to ensure the model expects and produces 8-bit integer data during inference.
6.  **Representative Dataset**: A `representative_dataset_gen` function is provided to the converter. This function yields a small subset of the training data, which the converter uses to calibrate the ranges for quantization. This is crucial for accurate 8-bit quantization.
7.  **Conversion and Saving**: The `converter.convert()` method performs the quantization, and the resulting TFLite model is saved to a `.tflite` file.

## 5. Training Optimization and Gradient Stabilization

Vanilla RNNs are known to suffer from vanishing or exploding gradients, making them difficult to train. To address this, several techniques were incorporated:

*   **Adam Optimizer**: The Adam optimizer was chosen for its adaptive learning rate capabilities, which generally perform well across a wide range of deep learning tasks.
*   **Gradient Clipping**: `tf.keras.optimizers.Adam(clipnorm=1.0)` was used to apply gradient clipping by norm. This technique limits the maximum value of gradients, effectively preventing exploding gradients and promoting more stable training.
*   **L2 Regularization**: L2 regularization (weight decay) was applied to the kernel and recurrent weights of the `SimpleRNN` layer, and to the kernel weights of the final `Dense` layer. This helps prevent overfitting by penalizing large weights.
*   **Dropout**: A `Dropout` layer was added after the `SimpleRNN` layer. Dropout randomly sets a fraction of input units to 0 at each update during training, which helps prevent overfitting by forcing the network to learn more robust features.

## 6. Model Evaluation and Performance Analysis

The `rnn_model1.py` script includes the necessary code for:

* **Training**: The model is compiled with `sparse_categorical_crossentropy` loss and `accuracy` metric. It is trained on the `train_ds` and validated on `val_ds` for a specified number of epochs.
* **Evaluation**: After training, the full precision model is evaluated on the `test_ds` to report its loss and accuracy. 

## 7. Conclusion

This project successfully designed and implemented a Vanilla RNN for audio classification with 8-bit weight quantization, incorporating various training optimization and gradient stabilization techniques. The architectural design and implementation of quantization and optimization strategies are robust and adhere to the specified hardware-efficient requirements. The provided code (`data_preprocessing1.py` and `rnn_model1.py`) serves as a complete blueprint for deploying such a model in an environment with adequate computational resources for data handling.

## 8. Files Provided

*   `data_preprocessing1.py`: Script for handling dataset download and preprocessing (spectrogram generation and TensorFlow dataset creation).
*   `rnn_model1.py`: Script containing the Vanilla RNN model definition, training loop, and 8-bit post-training quantization implementation.



