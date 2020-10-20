# -*- coding: utf-8 -*-
# @Time    : 2020/10/15 1:47 下午
# @Author  : jeffery
# @FileName: weibo_trainer.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from utils import inf_loop, MetricTracker
from base import BaseTrainer
import torch
import numpy as np
import time


class Trainer(BaseTrainer):
    """
    Trainer class
    """

    def __init__(self, model, criterion, metric_ftns, optimizer, config, data_loader,
                 valid_data_loader=None, test_data_loader=None, lr_scheduler=None, len_epoch=None):
        super().__init__(model, criterion, metric_ftns, optimizer, config)
        self.config = config
        self.data_loader = data_loader
        if len_epoch is None:
            # epoch-based training
            self.len_epoch = len(self.data_loader)
        else:
            # iteration-based training
            self.data_loader = inf_loop(data_loader)
            self.len_epoch = len_epoch
        self.valid_data_loader = valid_data_loader
        self.test_data_loader = test_data_loader
        self.do_validation = self.valid_data_loader is not None
        self.do_inference = self.test_data_loader is not None
        self.lr_scheduler = lr_scheduler
        self.log_step = int(np.sqrt(data_loader.batch_size))

        self.train_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.valid_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)
        self.test_metrics = MetricTracker('loss', *[m.__name__ for m in self.metric_ftns], writer=self.writer)

    def _train_epoch(self, epoch):
        """
        Training logic for an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains average loss and metric in this epoch.
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            input_ids, attention_masks, text_lengths, labels = data

            if 'cuda' == self.device.type:
                input_ids = input_ids.cuda()
                if attention_masks is not None:
                    attention_masks = attention_masks.cuda()
                text_lengths = text_lengths.cuda()
                labels = labels.cuda()
            preds, embedding = self.model(input_ids, attention_masks, text_lengths)
            preds = preds.squeeze()
            loss = self.criterion[0](preds, labels)
            loss.backward()
            self.optimizer.step()
            self.writer.set_step((epoch - 1) * self.len_epoch + batch_idx)
            self.train_metrics.update('loss', loss.item())

            for met in self.metric_ftns:
                self.train_metrics.update(met.__name__, met(preds, labels))

            if batch_idx % self.log_step == 0:
                self.logger.debug('Train Epoch: {} {} Loss: {:.3f}'.format(epoch, self._progress(batch_idx),
                                                                           loss.item()))

            if batch_idx == self.len_epoch:
                break
        log = self.train_metrics.result()
        if self.do_validation:
            val_log = self._valid_epoch(epoch)
            log.update(**{'val_' + k: v for k, v in val_log.items()})

        if self.do_inference:
            test_log = self._inference_epoch(epoch)
            log.update(**{'test_' + k: v for k, v in test_log.items()})

        if self.lr_scheduler is not None:
            self.lr_scheduler.step()
        return log

    def _valid_epoch(self, epoch):
        """
        Validate after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.valid_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.valid_data_loader):
                input_ids, attention_masks, text_lengths, labels = data

                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda()
                    if attention_masks is not None:
                        attention_masks = attention_masks.cuda()
                    text_lengths = text_lengths.cuda()
                    labels = labels.cuda()
                preds, embedding = self.model(input_ids, attention_masks, text_lengths)
                preds = preds.squeeze()
                if self.add_graph:
                    input_model = self.model.module if (len(self.config.config['device_id']) > 1) else self.model
                    self.writer.writer.add_graph(input_model,
                                                 [input_ids, attention_masks, text_lengths])
                    self.add_graph = False
                loss = self.criterion[0](preds, labels)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'valid')
                self.valid_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.valid_metrics.update(met.__name__, met(preds, labels))

        # add histogram of model parameters to the tensorboard
        for name, p in self.model.named_parameters():
            self.writer.add_histogram(name, p, bins='auto')
        return self.valid_metrics.result()

    def _inference_epoch(self, epoch):
        """
        Inference after training an epoch

        :param epoch: Integer, current training epoch.
        :return: A log that contains information about validation
        """
        self.model.eval()
        self.test_metrics.reset()
        with torch.no_grad():
            for batch_idx, data in enumerate(self.test_data_loader):
                input_ids, attention_masks, text_lengths, labels = data

                if 'cuda' == self.device.type:
                    input_ids = input_ids.cuda()
                    if attention_masks is not None:
                        attention_masks = attention_masks.cuda()
                    text_lengths = text_lengths.cuda()
                    labels = labels.cuda()
                preds, embedding = self.model(input_ids, attention_masks, text_lengths)
                preds = preds.squeeze()
                loss = self.criterion[0](preds, labels)

                self.writer.set_step((epoch - 1) * len(self.valid_data_loader) + batch_idx, 'test')
                self.test_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(preds, labels))

                # add histogram of model parameters to the tensorboard
            for name, p in self.model.named_parameters():
                self.writer.add_histogram(name, p, bins='auto')
            return self.test_metrics.result()

    def _progress(self, batch_idx):
        base = '[{}/{} ({:.0f}%)]'
        if hasattr(self.data_loader, 'n_samples'):
            current = batch_idx * self.data_loader.batch_size
            total = self.data_loader.n_samples
        else:
            current = batch_idx
            total = self.len_epoch
        return base.format(current, total, 100.0 * current / total)
