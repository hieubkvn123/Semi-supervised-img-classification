import numpy as np
import tensorflow as tf

from argparse import ArgumentParser
from dataloader import DataLoader
from trainer import PiModel
from models import build_cls_model

# Define the argument parser and script's arguments
parser = ArgumentParser()
parser.add_argument('--input-1', type=str, required=True, help='Path to training data folder (Annotated)')
parser.add_argument('--input-2', type=str, required=True, help='Path to training data folder (Unannotated)')
parser.add_argument('--input-size', type=int, required=False, default=64, help='Input image size')
args = vars(parser.parse_args())

# Some constants
input_dim = (args['input_size'], args['input_size'], 3)

# 1. Build the model
print('[INFO] Building the model...')
model = build_cls_model(input_dim)
print(model.summary())

# 2. Build the dataloader
print('[INFO] Building data loader for supervised and unsupervised datasets...')
loader1 = DataLoader(args['input_1'], one_hot=True, augment=True, labels_as_subdir=True, batch_size=32)
loader2 = DataLoader(args['input_2'], one_hot=True, augment=True, labels_as_subdir=False, batch_size=32)

# 3. Build the trainer
print('[INFO] Building the trainer...')
trainer = PiModel(model, loader1, loader2)
print(trainer)

# 4. Start training
print('[INFO] Training starting...')
trainer.train()
