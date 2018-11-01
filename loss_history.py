# -*- coding: utf-8 -*-
# @Author: LogicJake
# @Date:   2018-10-31 10:45:22
# @Last Modified time: 2018-11-01 09:38:47
# 写一个LossHistory类，保存loss和acc

import keras
import matplotlib.pyplot as plt


class LossHistory(keras.callbacks.Callback):

    def on_train_begin(self, logs={}):
        self.losses = {'batch': [], 'epoch': []}

        self.val_loss = {'batch': [], 'epoch': []}
        self.val_output_mode_loss = {'batch': [], 'epoch': []}
        self.val_output_machine_loss = {'batch': [], 'epoch': []}
        self.val_output_anti_type_loss = {'batch': [], 'epoch': []}
        self.val_output_anti_first_loss = {'batch': [], 'epoch': []}
        self.val_output_anti_add_loss = {'batch': [], 'epoch': []}

        self.val_output_mode_acc = {'batch': [], 'epoch': []}
        self.val_output_machine_acc = {'batch': [], 'epoch': []}
        self.val_output_anti_type_acc = {'batch': [], 'epoch': []}
        self.val_output_anti_first_acc = {'batch': [], 'epoch': []}
        self.val_output_anti_add_acc = {'batch': [], 'epoch': []}

    def on_batch_end(self, batch, logs={}):
        self.losses['batch'].append(logs.get('loss'))
        self.val_loss['batch'].append(logs.get('val_loss'))

    def on_epoch_end(self, batch, logs={}):
        self.val_loss['epoch'].append(logs.get('val_loss'))
        self.val_output_mode_loss['epoch'].append(
            logs.get('val_output_mode_loss'))
        self.val_output_machine_loss['epoch'].append(
            logs.get('val_output_machine_loss'))
        self.val_output_anti_type_loss['epoch'].append(
            logs.get('val_output_anti_type_loss'))
        self.val_output_anti_first_loss['epoch'].append(
            logs.get('val_output_anti_first_loss'))
        self.val_output_anti_add_loss[
            'epoch'].append(logs.get('val_output_anti_add_loss'))

        self.val_output_mode_acc['epoch'].append(
            logs.get('val_output_mode_acc'))
        self.val_output_machine_acc['epoch'].append(
            logs.get('val_output_machine_acc'))
        self.val_output_anti_type_acc['epoch'].append(
            logs.get('val_output_anti_type_acc'))
        self.val_output_anti_first_acc['epoch'].append(
            logs.get('val_output_anti_first_acc'))
        self.val_output_anti_add_acc['epoch'].append(
            logs.get('val_output_anti_add_acc'))

    def loss_plot(self, loss_type):
        iters = range(len(self.val_loss[loss_type]))
        plt.figure()

        if loss_type == 'epoch':
            # val_loss

            plt.subplot(2, 2, 1)
            plt.plot(iters, self.val_loss[
                     loss_type], label='val loss')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('loss')
            plt.legend(loc="upper right")

            plt.subplot(2, 2, 2)
            plt.plot(iters, self.val_output_mode_loss[
                     loss_type], label='mode loss')
            plt.plot(iters, self.val_output_machine_loss[
                     loss_type],  label='machine loss')
            plt.plot(iters, self.val_output_anti_type_loss[
                     loss_type],  label='anti_type loss')
            plt.plot(iters, self.val_output_anti_first_loss[
                     loss_type],  label='anti_first loss')
            plt.plot(iters, self.val_output_anti_add_loss[
                     loss_type], label='anti_add loss')

            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('loss')
            plt.legend(loc="upper right")

            plt.subplot(2, 1, 2)
            plt.plot(iters, self.val_output_mode_acc[
                     loss_type], label='mode acc')
            plt.plot(iters, self.val_output_machine_acc[
                     loss_type], label='machine acc')
            plt.plot(iters, self.val_output_anti_type_acc[
                     loss_type], label='anti_type acc')
            plt.plot(iters, self.val_output_anti_first_acc[
                     loss_type], label='anti_first acc')
            plt.plot(iters, self.val_output_anti_add_acc[
                     loss_type], label='anti_add acc')
            plt.grid(True)
            plt.xlabel(loss_type)
            plt.ylabel('acc')
            plt.legend(loc="upper right")
        plt.savefig('result/loss-acc.png')
