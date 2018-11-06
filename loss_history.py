# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-31 10:45:22
# @Last Modified time: 2018-11-06 10:28:34
# 写一个LossHistory类，保存loss和acc

import keras
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.loss = {'batch': [], 'epoch': []}
        self.val_loss = {'batch': [], 'epoch': []}
        self.val_output_mm_loss = {'batch': [], 'epoch': []}
        self.val_output_anti_loss = {'batch': [], 'epoch': []}
        self.val_output_anti_add_loss = {'batch': [], 'epoch': []}

        self.val_output_mm_acc = {'batch': [], 'epoch': []}
        self.val_output_anti_acc = {'batch': [], 'epoch': []}
        self.val_output_anti_add_acc = {'batch': [], 'epoch': []}
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
        self.val_output_mm_loss['epoch'].append(
            logs.get('val_output_mm_loss'))
        self.val_output_anti_loss['epoch'].append(
            logs.get('val_output_anti_loss'))
        self.val_output_anti_add_loss[
            'epoch'].append(logs.get('val_output_anti_add_loss'))

        self.val_output_mm_acc['epoch'].append(
            logs.get('val_output_mm_acc'))
        self.val_output_anti_acc['epoch'].append(
            logs.get('val_output_anti_acc'))
        self.val_output_anti_add_acc['epoch'].append(
            logs.get('val_output_anti_add_acc'))

        self.acc['epoch'].append(
            logs.get('acc'))
        self.val_acc['epoch'].append(
            logs.get('val_acc'))

    def loss_plot(self, folder, loss_type):
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
