# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-31 10:45:22
# @Last Modified time: 2018-11-14 10:52:44
# LossHistory类，保存loss和acc

import matplotlib.pyplot as plt
import os
from keras.callbacks import Callback


class LossHistory(Callback):

    def on_train_begin(self, logs={}):
        self.loss = []
        self.val_loss = []
        self.mm_loss = []
        self.mm_val_loss = []
        self.anti_loss = []
        self.anti_val_loss = []

        self.mm_acc = []
        self.mm_val_acc = []
        self.anti_acc = []
        self.anti_val_acc = []

    def on_epoch_end(self, batch, logs={}):
        self.loss.append(logs.get('loss'))
        self.val_loss.append(logs.get('val_loss'))
        self.mm_loss.append(logs.get('mm_loss'))
        self.mm_val_loss.append(logs.get('val_mm_loss'))
        self.anti_loss.append(logs.get('anti_loss'))
        self.anti_val_loss.append(logs.get('val_anti_loss'))

        self.mm_acc.append(logs.get('mm_acc'))
        self.mm_val_acc.append(logs.get('val_mm_acc'))
        self.anti_acc.append(logs.get('anti_acc'))
        self.anti_val_acc.append(logs.get('val_anti_acc'))

    def save_loss(self, plot, folder):
        if plot:
            iters = range(len(self.val_loss))  # x's range
            plt.figure()

            # val_loss and loss
            plt.plot(iters, self.loss, label='loss')
            plt.plot(iters, self.val_loss, label='val loss')
            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc="upper right")
            plt.savefig(folder + 'entire_loss.png')

            # mm acc and loss
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(iters, self.mm_acc, label='acc')
            plt.plot(iters, self.mm_val_acc, label='val acc')
            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.legend(loc="upper right")

            plt.subplot(2, 1, 2)
            plt.plot(iters, self.mm_loss, label='loss')
            plt.plot(iters, self.mm_val_loss, label='val loss')
            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc="upper right")
            plt.savefig(folder + 'mm' + os.path.sep + 'loss_acc.png')

            # anti acc and loss
            plt.clf()
            plt.subplot(2, 1, 1)
            plt.plot(iters, self.anti_acc, label='acc')
            plt.plot(iters, self.anti_val_acc, label='val acc')
            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('acc')
            plt.legend(loc="upper right")

            plt.subplot(2, 1, 2)
            plt.plot(iters, self.anti_loss, label='loss')
            plt.plot(iters, self.anti_val_loss, label='val loss')
            plt.grid(True)
            plt.xlabel('epoch')
            plt.ylabel('loss')
            plt.legend(loc="upper right")
            plt.savefig(folder + 'anti' + os.path.sep + 'loss_acc.png')

        members = {}
        members['loss'] = self.loss
        members['val_loss'] = self.val_loss
        members['mm_loss'] = self.mm_loss
        members['mm_val_loss'] = self.mm_val_loss
        members['anti_loss'] = self.anti_loss
        members['anti_val_loss'] = self.anti_val_loss
        members['mm_acc'] = self.mm_acc
        members['mm_val_acc'] = self.mm_val_acc
        members['anti_acc'] = self.anti_acc
        members['anti_val_acc'] = self.anti_val_acc

        with open(folder + 'res.txt', 'w') as outfile:
            outfile.write(str(members))
