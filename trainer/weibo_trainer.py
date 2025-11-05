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
        """
        self.model.train()
        self.train_metrics.reset()
        for batch_idx, data in enumerate(self.data_loader):
            self.optimizer.zero_grad()
            input_ids, attention_masks, text_lengths, labels,_ = data
            input_ids = input_ids.to(self.device)
            if attention_masks is not None:
                attention_masks = attention_masks.to(self.device)
            text_lengths = text_lengths.to(self.device)
            labels = labels.to(self.device)
            preds, embedding = self.model(input_ids, attention_masks, text_lengths)
            preds = preds.squeeze()
            loss = self.criterion[0](preds, labels)

            if batch_idx % 50 == 0: # 每 50 个 batch 打印一次
                # 使用我们之前修复的 argmax 逻辑来获取预测类别
                pred_classes = torch.argmax(preds, dim=1) 
                correct_count = (pred_classes == labels).sum().item()
                
                print(f"\n--- 调试信息 (Batch {batch_idx}) ---")
                print(f"模型原始输出 (preds) (前5个): \n{preds[:5]}")
                print(f"模型预测类别 (pred_classes) (前5个): {pred_classes[:5]}")
                print(f"真实标签 (labels) (前5个): {labels[:5]}")
                print(f"本批次正确个数: {correct_count} / {len(labels)}")
                print(f"本批次损失 (Loss): {loss.item()}")
                print("-----------------\n")
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
            log.update(**{'val_'+k: v for k, v in val_log.items()})

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
                input_ids, attention_masks, text_lengths, labels, _ = data

                # --- 使用 .to(self.device) 简化了设备转移 ---
                input_ids = input_ids.to(self.device)
                if attention_masks is not None:
                    attention_masks = attention_masks.to(self.device)
                text_lengths = text_lengths.to(self.device)
                labels = labels.to(self.device)
                
                preds, embedding = self.model(input_ids, attention_masks, text_lengths)
                preds = preds.squeeze()

                # --- 修复：修复了 add_graph 不能处理 None 的问题 ---
                if self.add_graph: # 检查 self.add_graph (在你的代码中是 batch_idx == 0 ...)
                    # 1. 准备模型 (处理 DataParallel)
                    input_model = self.model.module if (len(self.config.config['device_id']) > 1) else self.model
                    
                    # 2. 准备示例输入，处理 attention_masks 为 None 的情况
                    graph_input_attn_mask = attention_masks
                    if attention_masks is None:
                        # 如果 attention_mask 是 None, 创建一个假的、全 1 的张量
                        # 它的形状、类型和设备都必须和 input_ids 一致
                        graph_input_attn_mask = torch.ones_like(input_ids)
                    
                    graph_inputs = [input_ids, graph_input_attn_mask, text_lengths]
                    
                    # 3. 调用 add_graph
                    self.writer.writer.add_graph(input_model, graph_inputs)
                    self.add_graph = False
                # --- 修复结束 ---

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
                # --- 修复 1：解包 5 个元素 ---
                input_ids, attention_masks, text_lengths, labels, _ = data

                # --- 修复 4：统一使用 .to(self.device) ---
                input_ids = input_ids.to(self.device)
                if attention_masks is not None:
                    attention_masks = attention_masks.to(self.device)
                text_lengths = text_lengths.to(self.device)
                labels = labels.to(self.device)
                
                preds, embedding = self.model(input_ids, attention_masks, text_lengths)
                preds = preds.squeeze()
                loss = self.criterion[0](preds, labels)

                # --- 修复 3：使用 test_data_loader 的长度 ---
                self.writer.set_step((epoch - 1) * len(self.test_data_loader) + batch_idx, 'test')
                self.test_metrics.update('loss', loss.item())
                for met in self.metric_ftns:
                    self.test_metrics.update(met.__name__, met(preds, labels))

        # --- 修复 2：修正缩进，将下面两块代码移出 'with' 块 ---
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
