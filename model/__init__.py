# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 2:52 下午
# @Author  : jeffery
# @FileName: __init__.py.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
import torch
import transformers
import model.models as module_models
import model.loss as module_loss
import model.metric as module_metric

__all__ = ["makeModel", "makeLoss", "makeMetrics", "makeOptimizer","makeLrSchedule"]


def makeModel(config):
    return config.init_obj('model_arch', module_models,)


def makeLoss(config):
    return [getattr(module_loss, crit) for crit in config['loss']]


def makeMetrics(config):
    return [getattr(module_metric, met) for met in config['metrics']]


def makeOptimizer(config, model):
    parameters = []
    model_parameters = [*filter(lambda p: p.requires_grad, model.parameters())]
    if 'transformer' in config.config['model_arch']['type'].lower():
        transformers_parameters = [*filter(lambda p: p.requires_grad, model.transformer_model.parameters())]
        model_parameters = list(set(model_parameters)-set(transformers_parameters))
        parameters.append({
            'params':transformers_parameters,
            'weight_decay':0.,
            'lr': 2e-5
        })
    parameters.append({
        'params': model_parameters,
        'weight_decay': 0.,
        'lr': 1e-3
    })
    optimizer = config.init_obj('optimizer', torch.optim, parameters)

    return optimizer


def makeLrSchedule(config, optimizer, train_set):
    # lr_scheduler = config.init_obj('lr_scheduler', optimization.lr_scheduler, optimizer)
    lr_scheduler = config.init_obj('lr_scheduler', transformers.optimization, optimizer,
                                   num_training_steps=int(len(train_set) / train_set.batch_size))
    return lr_scheduler

