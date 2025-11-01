import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, regularizers
import os
import numpy as np

from data_preprocessing1 import create_dataset_from_filepaths, LABELS
# from generate_report import generate_model_report, generate_model_report_pdf

class VanillaRNN(models.Model):
    def __init__(self, rnn_units, num_classes, l2_reg=0.001, dropout_rate=0.2):
        super(VanillaRNN, self).__init__()
        self.rnn = layers.SimpleRNN(
            rnn_units,
            activation="tanh",
            return_sequences=False,
            kernel_regularizer=regularizers.l2(l2_reg),
            recurrent_regularizer=regularizers.l2(l2_reg),
            unroll=True
        )
        self.dropout = layers.Dropout(dropout_rate)
        self.classifier = layers.Dense(
            num_classes,
            activation="softmax",
            kernel_regularizer=regularizers.l2(l2_reg)
        )

    def call(self, inputs):
        x = self.rnn(inputs)
        x = self.dropout(x)
        x = self.classifier(x)
        return x

if __name__ == '__main__':
    BATCH_SIZE = 32

    train_ds, val_ds, test_ds, label_map = create_dataset_from_filepaths(batch_size=BATCH_SIZE)

    if train_ds is None:
        print("Failed to load dataset. Exiting.")
    else:
        # Explicitly defined the input shape based on preprocessing logic
        # sequence_length = (SAMPLE_RATE - N_FFT) // HOP_LENGTH + 1 = (16000 - 1024) // 256 + 1 = 59
        # features = N_FFT // 2 + 1 = 1024 // 2 + 1 = 513
        concrete_input_shape = (59, 513)

        rnn_units = 256
        num_classes = len(LABELS)
        epochs = 20
        model_path = "full_precision_rnn_model.h5"

        if os.path.exists(model_path):
            print(f"Loading existing model from {model_path}")
            model = models.load_model(model_path)
        else:
            model = models.Sequential([
                layers.Input(shape=concrete_input_shape),
                layers.SimpleRNN(
                    rnn_units,
                    activation="tanh",
                    return_sequences=False,
                    kernel_regularizer=regularizers.l2(0.001),
                    recurrent_regularizer=regularizers.l2(0.001),
                    unroll=True
                ),
                layers.Dropout(0.3),
                layers.Dense(
                    num_classes,
                    activation="softmax",
                    kernel_regularizer=regularizers.l2(0.001)
                )
            ])
            model.summary()

            initial_learning_rate = 0.0001
            optimizer = tf.keras.optimizers.Adam(learning_rate=initial_learning_rate, clipnorm=1.0)

            model.compile(optimizer=optimizer, loss="sparse_categorical_crossentropy", metrics=["accuracy"])

            reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
                monitor='val_loss',
                factor=0.5,
                patience=3,
                min_lr=1e-7,
                verbose=1
            )

            print("\nTraining the full precision model...")
            history = model.fit(
                train_ds,
                validation_data=val_ds,
                epochs=epochs,
                callbacks=[reduce_lr]
            )

            print("\nEvaluating the full precision model on the test set...")
            loss, accuracy = model.evaluate(test_ds)
            print(f"Test Loss: {loss:.4f}")
            print(f"Test Accuracy: {accuracy:.4f}")

            model.save(model_path)
            print(f"Full precision model saved to {model_path}")

            # Commented because I used these functions for report generation
            """generate_model_report(model, history, train_ds, val_ds, test_ds, LABELS, report_name="vanilla_rnn_report")
            generate_model_report_pdf(model, history, train_ds, val_ds, test_ds, LABELS)"""

        #################################################################################################################

        print("\nApplying post-training 8-bit quantization...")

        def representative_dataset_gen():
            for input_value, _ in train_ds.unbatch().take(100):
                yield [tf.expand_dims(input_value, axis=0)]

        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS, tf.lite.OpsSet.SELECT_TF_OPS]
        converter.inference_input_type = tf.int8
        converter.inference_output_type = tf.int8
        converter.representative_dataset = representative_dataset_gen
        converter._experimental_lower_tensor_list_ops = False

        tflite_model = converter.convert()

        tflite_model_path = "quantized_rnn_model.tflite"
        with open(tflite_model_path, "wb") as f:
            f.write(tflite_model)
        print(f"Quantized TFLite model saved to {tflite_model_path}")


