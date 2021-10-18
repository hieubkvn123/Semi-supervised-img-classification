import os
import tqdm
import numpy as np
import pandas as pd

### Tensorflow dependencies ###
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.models import Model
from tensorflow.keras import backend as K
from tensorflow.keras.losses import MeanSquaredError, BinaryCrossentropy

class PiModel:
    def __init__(self, model, supervised_loader, unsupervised_loader, model_name='pi_model',
            save_path='checkpoints', epochs=100, lr=0.0001, steps_per_epoch=50, 
            val_steps=10, save_steps=5, n_labelled=100, n_total=60000, callbacks=None):
        self.model = model
        self.train_dataset = supervised_loader.get_train_dataset()
        self.u_dataset = unsupervised_loader.get_train_dataset()
        self.val_dataset = supervised_loader.get_val_dataset()
        self.steps_per_epoch = supervised_loader.get_steps_per_epoch()
        self.val_steps = supervised_loader.get_val_steps()
        self.unsupervised_steps = unsupervised_loader.get_steps_per_epoch()
        self.model_name = model_name
        self.save_path = save_path
        self.epochs = epochs
        self.lr = lr
        self.steps_per_epoch = steps_per_epoch
        self.val_steps = val_steps
        self.save_steps = save_steps
        self.callbacks = callbacks
        self.bce = BinaryCrossEntropy(from_logits=False)
        self.mse = MeanSquaredError()

        ### For sigmoid rampup weight function ###
        self.Tmax = 80
        self.max_val = 30
        self.n_labelled = n_labelled
        self.n_total = n_total

    def sigmoid_rampup(self, t):
        max_val = self.max_val * (self.n_labelled / self.n_total)

        if(t == 0):
            return 0
        elif(epoch >= max_epochs):
            return max_val

        return max_val * np.exp(-5 * (1 - t/self.Tmax) ** 2)

    @tf.function
    def train_step(model, opt, batch, epoch):
        with tf.GradientTape() as tape:
            weak_aug, strong_aug, labels = batch

            # 1. Calculate the classification loss
            predictions, logits_weak = model(weak_aug, training=True)
            cls_loss = self.bce(labels, predictions)

            # 2. Calculate the consistency loss
            predictions, logits_strong = model(strong_aug, training=True)
            consistency_loss = self.mse(logits_weak, logits_strong)

            # 3. Overall loss
            consistency_weight = self.sigmoid_rampup(epoch)
            loss = cls_loss + consistency_weight * consistency_loss

            # 4. Gradients and optimization
            gradients = tape.gradient(loss, model.trainable_variables)

        opt.apply_gradients(zip(gradients, model.trainable_variables))
        return loss

    @tf.function
    def train_step_unsupervised(model, opt, batch, epoch):
        with tf.GradientTape() as tape:
            weak_aug, strong_aug = batch

            predictions, logits_weak = model(weak_aug, training=True)
            predictions, logits_strong = model(strong_aug, training=True)

            consistency_weight = self.sigmoid_rampup(epoch)
            loss = consistency_weight * self.mse(logits_weak, logits_strong)

            gradients = tape.gradient(loss, model.trainable_variables)
        opt.apply_gradients(zip(gradients, model.trainable_variables))

        return loss

    @tf.function
    def val_step(model, batch):
        weak_aug, strong_aug, labels = batch
        predictions, logits = model(images, training=False)
        loss = self.bce(labels, predictions)

        return loss

    def train(self):
        if(not os.path.exists(self.save_path)):
            os.mkdir(self.save_path)
            print('[INFO] Checkpoint path created ...')
            self.model.save(os.path.join(self.save_path, self.model_name, 'h5'))            
    
        optimizer = Adam(lr=self.lr, amsgrad=True)
        
        train_losses = []
        val_losses = []
        for i in range(self.epochs):
            print(f'Epoch #[{i+1}/{epochs}]')

            ### Run through train supervised data directory ###
            with tqdm.tqdm(total=self.steps_per_epoch) as pbar:
                for batch in self.train_dataset:
                    train_loss = self.train_step(self.model, optimizer, batch, step=i+1)
                    train_loss = train_loss.numpy()

                    train_losses.append(train_loss)
                    pbar.set_postfix({'train_loss' : f'{train_loss:.4f}'})
                    pbar.update(1)

            ### Run through unsupervised data directory ###
            if(self.u_dataset is not None):
                with tqdm.tqdm(total=self.unsupervised_steps, colour='red') as pbar:
                    for batch in self.u_dataset:
                        unsupervised_loss = self.train_step_unsupervised(self.model, optimizer, batch, i+1)
                        global_step += 1
                        unsupervised_loss = unsupervised_loss.numpy()

                        pbar.set_postfix({'unsupervised_loss' : f'{unsupervised_loss:.4f}'})
                        pbar.update(1)

            ### Run through val supervised data directory ###
            with tqdm.tqdm(total=self.val_steps, colour='green') as pbar:
                for batch in val_dataset:
                    val_loss = self.val_step(self.model, batch)
                    val_loss = val_loss.numpy()

                    val_losses.append(val_loss)
                    pbar.set_postfix({'val_loss' : f'{val_loss:.4f}'})
                    pbar.update(1)

            # Compute mean losses
            mean_train_loss = np.array(train_losses).mean()
            mean_val_loss = np.array(val_losses).mean()

            # Save models
            if((i + 1) % self.save_steps == 0):
                print('[INFO] Checkpointing weights ...')
                self.model.save_weights(os.path.join(self.save_path, self.model_name, 'hdf5')

