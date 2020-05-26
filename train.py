import argparse
import collections
import torch
import numpy as np
# import data_process.data_loaders as module_data
from data_process import data_process as module_data_process
from torch.utils.data import dataloader as module_data_loader
from base import base_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer, BertTrainer
import transformers as optimization
import os

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config, use_transformers):
    logger = config.get_logger('train')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup data_set, data_process instances
    dataset = config.init_obj('dataset', module_data_process, device=device)

    # build model architecture, then print to console
    model = config.init_obj('model_arch', module_arch, word_embedding=dataset.word_embedding)
    logger.info(model)

    # get function handles of loss and metrics
    criterion = [getattr(module_loss, crit) for crit in config['loss']]
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    if use_transformers:
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        if 'bert' in config.config['model_arch']['type'].lower():
            transformers_params = [*filter(lambda p: p.requires_grad, model.bert.parameters())]
            other_params = [*filter(lambda p: p.requires_grad,
                                    [param for name, param in model.named_parameters() if 'bert' not in name])]

        if 'xlnet' in config.config['model_arch']['type'].lower():
            transformers_params = [*filter(lambda p: p.requires_grad, [param for name, param in model.named_parameters() if 'xlnet'  in name])]
            other_params = [*filter(lambda p: p.requires_grad,
                                [param for name, param in model.named_parameters() if 'xlnet' not in name])]
        # bert的优化器略有不同
        train_dataloader = config.init_obj('data_loader', module_data_loader, dataset=dataset.train_set,
                                           collate_fn=dataset.bert_collate_fn)
        valid_dataloader = config.init_obj('data_loader', module_data_loader, dataset=dataset.test_set,
                                           collate_fn=dataset.bert_collate_fn)

        optimizer = config.init_obj('optimizer', optimization, [
            {"params": transformers_params, 'lr': 5e-5, "weight_decay": 0},
            {"params": other_params, 'lr': 1e-3, "weight_decay": 0},
        ])
        lr_scheduler = config.init_obj('lr_scheduler', optimization.optimization, optimizer,
                                       num_training_steps=int(
                                           len(train_dataloader.dataset) / train_dataloader.batch_size))
    else:
        # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
        trainable_params = [*filter(lambda p: p.requires_grad, model.parameters())]
        train_dataloader = config.init_obj('data_loader', module_data_loader, dataset=dataset.train_set,
                                           collate_fn=dataset.collate_fn)
        valid_dataloader = config.init_obj('data_loader', module_data_loader, dataset=dataset.test_set,
                                           collate_fn=dataset.collate_fn)
        optimizer = config.init_obj('optimizer', torch.optim, trainable_params)
        lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      train_iter=train_dataloader,
                      valid_iter=valid_dataloader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


def run(config_file):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0,1', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    print(config.config['model_arch']['type'].lower())

    # 是否使用bert作为预训练模型
    if 'bert' in config.config['model_arch']['type'].lower() or 'xlnet' in config.config['model_arch']['type'].lower():
        main(config, use_transformers=True)
    else:
        main(config, use_transformers=False)


if __name__ == '__main__':
    # ---------------------------------------微博情感二分类-------------------------------------------------
    # run('configs/binary_classification/fast_text_config.json')
    # run('configs/binary_classification/text_cnn_config.json')
    # run('configs/binary_classification/text_cnn_1d_config.json')
    # run('configs/binary_classification/rnn_config.json')
    # run('configs/binary_classification/lstm_config.json')
    # run('configs/binary_classification/gru_config.json')
    # run('configs/binary_classification/rcnn_config.json')
    # run('configs/binary_classification/rnn_attention_config.json')
    # run('configs/binary_classification/dpcnn_config.json')
    # run('configs/binary_classification/bert_config.json')
    # run('configs/binary_classification/bert_rnn_config.json')
    # run('configs/binary_classification/bert_rcnn_config.json')
    # run('configs/binary_classification/bert_cnn_config.json')
    # ---------------------------------------cnews 十分类-------------------------------------------------

    run('configs/multi_classification/fast_text_config.json')
    run('configs/multi_classification/text_cnn_config.json')
    run('configs/multi_classification/text_cnn_1d_config.json')
    run('configs/multi_classification/rnn_config.json')
    run('configs/multi_classification/lstm_config.json')
    run('configs/multi_classification/gru_config.json')
    run('configs/multi_classification/rcnn_config.json')
    run('configs/multi_classification/rnn_attention_config.json')
    run('configs/multi_classification/dpcnn_config.json')
    run('configs/multi_classification/han_config.json')
    # run('configs/multi_classification/xlnet_config.json')
