import pandas as pd
import logging
import random

import model.net
import matplotlib.pyplot as plt
from utils.utils import get_scaler, transform_data
from tensorflow.keras.optimizers import SGD, Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint


class Trainer:
    def __init__(self, cfg):
        self.cfg = cfg
        logging.info(f'--> train network: {self.cfg.network}')
        self.build_dataset()
        self.build_model()
        self.build_optim()

    def build_dataset(self):
        data_name = f"{self.cfg.model.encoder_dim}_{self.cfg.data.pred_steps}"
        logging.info(f'--> building dataset from: {data_name}')
        self.trainDF = pd.read_csv(self.cfg.data.root + f"train{data_name}.csv")
        self.testDF = pd.read_csv(self.cfg.data.root + f"test{data_name}.csv")
        logging.info(f'--> train data shape: {self.trainDF.shape}')
        logging.info(f'--> test data shape: {self.testDF.shape}')
        fips = self.trainDF['fips'].unique().tolist()
        random.shuffle(fips)
        train_fips = fips[:int(len(fips)*0.8)]  # 80% counties for training, the other for validation 
        val_fips = fips[int(len(fips)*0.8):]    # and test data are the last pred_steps days for all counties.
        traindf = self.trainDF[self.trainDF['fips'].isin(train_fips)]   
        valdf = self.trainDF[self.trainDF['fips'].isin(val_fips)]
        self.scaler_covid = get_scaler('covid', fips, self.cfg)
        self.scaler_travel = get_scaler('travel', fips, self.cfg)
        self.train_data = transform_data(traindf, self.cfg, [self.scaler_covid, self.scaler_travel])
        self.val_data = transform_data(valdf, self.cfg, [self.scaler_covid, self.scaler_travel])

    def build_model(self):
        logging.info(f'--> building models: {self.cfg.network}')
        self.network = getattr(model.net, self.cfg.network)(self.cfg)
        #self.network.train_model.summary(print_fn=logging.info)

    def build_optim(self):
        early_stopping = EarlyStopping(patience=80, verbose=1)
        reduce_lr = ReduceLROnPlateau(factor=0.1, patience=40, min_lr=1e-5, verbose=1)
        checkpointer = ModelCheckpoint(filepath = self.cfg.weight_file, verbose=1, monitor='val_loss', 
                                    mode='auto', save_best_only=True)
        self.loss = 'mean_squared_error'
        if self.cfg.optim.optimizer == 'Adam':
            self.optimizer = Adam(learning_rate=self.cfg.train.lr)
        elif self.cfg.optim.optimizer == 'SGD':
            self.optimizer = SGD(learning_rate=self.cfg.train.lr, 
                        momentum=self.cfg.optim.momentum, decay=self.cfg.optim.weight_decay, nesterov=True)
        else:
            raise ValueError(f"{self.cfg.optim.optimizer} is not supported.")
        self.callbacks=[reduce_lr, checkpointer, early_stopping]
    
    def train(self):
        self.network.train_model.compile(loss=self.loss, optimizer=self.optimizer)
        history = self.network.train_model.fit(
            self.train_data[:-1], self.train_data[-1], validation_data=(self.val_data[:-1], self.val_data[-1]), 
            epochs=self.cfg.train.epochs, batch_size=self.cfg.train.batch_size, 
            shuffle=False, callbacks=self.callbacks, verbose=0
        )
        plt.plot(history.history['loss'], label = 'train')
        plt.plot(history.history['val_loss'], label = 'val')
        plt.legend()
        plt.savefig(fname=self.cfg.save_path + "/loss.png",figsize=[10,10])
        plt.close()
        logging.info("--> Finish Training.")
        logging.info(f"train loss: {history.history['loss'][-1]} | val loss: {history.history['val_loss'][-1]}.")

    def load_weights(self, weight_file):
        self.network.train_model.load_weights(weight_file)