import numpy as np
import pandas as pd

### Tensorflow dependencies ###
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

class PiModel:
    def __init__(self, model, train_dataset, val_datasaet, save_path='checkpoints', 
            epochs=100, lr=0.0001, steps_per_epoch=50, val_steps=10, save_steps=5, callbacks=None):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.save_path = save_path
        self.epochs = epochs
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.val_steps = val_steps
        self.save_steps = save_steps
        self.callbacks = callbacks


