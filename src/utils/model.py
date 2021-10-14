import logging
import tensorflow as tf
import time
import matplotlib.pyplot as plt
import pandas as pd
import os

def create_model(LOSSFUNCTION, OPTIMIZER, METRICS, NUM_ClASSES):
    LAYERS = [
          tf.keras.layers.Flatten(input_shape=(28, 28), name="inputLayer"),
          tf.keras.layers.Dense(300, activation="relu", name="hiddenLayer1"),
          tf.keras.layers.Dense(100, activation="relu", name="hiddenLayer2"),
          tf.keras.layers.Dense(10, activation="softmax", name="outputLayer"),
    ]
    model_clf = tf.keras.models.Sequential(LAYERS)
    model_clf.summary()
    logging.info(f"model creation successful")

    model_clf.compile(loss=LOSSFUNCTION, 
                  optimizer=OPTIMIZER, 
                  metrics=METRICS)
    logging.info("model compilation successful")
    return model_clf ## << untrained model

def get_unique_filename(filename):
    unique_filename = time.strftime(f"%Y%m%d_%H%M%S_{filename}")
    return unique_filename

def save_model(model, model_name, model_dir):
    unique_filename = get_unique_filename(model_name)
    path_to_model = os.path.join(model_dir, unique_filename)
    logging.info(f"model saved successfully at {path_to_model}")
    model.save(path_to_model)

def save_plot(loss_acc, plot_name, plot_dir):
    unique_filename = get_unique_filename(plot_name)
    path_to_plot = os.path.join(plot_dir, unique_filename)
    pd.DataFrame(loss_acc).plot(figsize=(10, 7))
    plt.grid(True)
    logging.info(f"plot saved successfully at {path_to_plot}")
    plt.savefig(path_to_plot)
