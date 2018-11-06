# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-31 10:45:22
# @Last Modified time: 2018-11-06 11:33:29
# 写一个LossHistory类，保存loss和acc

import keras
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}

        self.acc = {'batch': [], 'epoch': []}
        self.val_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.loss['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

        self.acc['batch'].append(logs.get('acc'))
        self.val_acc['batch'].append(logs.get('val_acc'))

    def on_epoch_end(self, batch, logs={}):
        self.loss['epoch'].append(logs.get('loss'))
        self.val_loss['epoch'].append(logs.get('val_loss'))

        self.acc['epoch'].append(logs.get('acc'))
        self.val_acc['epoch'].append(logs.get('val_acc'))

    def loss_plot(self, plot, folder, loss_type):
        if plot:
            iters = range(len(self.val_loss[loss_type]))
            plt.figure()

            if loss_type == 'epoch':
                # val_loss
                plt.subplot(2, 1, 1)
                plt.plot(iters, self.loss[
                         loss_type], label='loss')
                plt.plot(iters, self.val_loss[
                         loss_type], label='val loss')
                plt.grid(True)
                plt.xlabel(loss_type)
                plt.ylabel('loss')
                plt.legend(loc="upper right")

                plt.subplot(2, 1, 2)
                plt.plot(iters, self.acc[
                         loss_type], label='acc')
                plt.plot(iters, self.val_acc[
                         loss_type], label='val acc')
                plt.grid(True)
                plt.xlabel(loss_type)
                plt.ylabel('acc')
                plt.legend(loc="upper right")
            plt.savefig(folder + 'loss-acc.png')
        else:
            store_res = {}
            store_res['loss'] = self.loss
            store_res['val_loss'] = self.val_loss
            store_res['acc'] = self.acc
            store_res['val_acc'] = self.val_acc

            store_res = str(store_res)
            with open(folder + 'res.txt', 'w') as outfile:
                outfile.write(store_res)
