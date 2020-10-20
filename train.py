# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 4:46 下午
# @Author  : jeffery
# @FileName: train.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from utils import WordEmbedding
import torch
import numpy as np
from model import makeModel, makeLoss, makeMetrics, makeOptimizer, makeLrSchedule
from utils import ConfigParser
import yaml
import random

# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)
random.seed(SEED)


def main(config):
    from data_process import makeDataLoader
    # 针对不同的数据，训练过程的设置略有不同。
    # from trainer.weibo_trainer import Trainer # weibo
    # from trainer.cnews_trainer import Trainer # cnews
    from trainer.medical_question_trainer import Trainer

    logger = config.get_logger('train')
    train_dataloader, valid_dataloader, test_dataloader = makeDataLoader(config)

    model = makeModel(config)
    logger.info(model)

    criterion = makeLoss(config)
    metrics = makeMetrics(config)

    optimizer = makeOptimizer(config, model)
    lr_scheduler = makeLrSchedule(config, optimizer, train_dataloader.dataset)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=train_dataloader,
                      valid_data_loader=valid_dataloader,
                      test_data_loader=test_dataloader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


def run(config_fname):
    with open(config_fname, 'r', encoding='utf8') as f:
        config_params = yaml.load(f, Loader=yaml.Loader)
        config_params['config_file_name'] = config_fname

    config = ConfigParser.from_args(config_params)
    main(config)


if __name__ == '__main__':
    # run('configs/multilabel_classification/word_embedding_text_cnn.yml')
    # run('configs/multilabel_classification/word_embedding_text_cnn_1d.yml')
    # run('configs/multilabel_classification/word_embedding_fast_text.yml')
    # run('configs/multilabel_classification/word_embedding_rnn.yml')
    # run('configs/multilabel_classification/word_embedding_rcnn.yml')
    # run('configs/multilabel_classification/word_embedding_rnn_attention.yml')
    # run('configs/multilabel_classification/word_embedding_dpcnn.yml')

    run('configs/multilabel_classification/transformers_pure.yml')
    run('configs/multilabel_classification/transformers_cnn.yml')
    run('configs/multilabel_classification/transformers_rnn.yml')
    run('configs/multilabel_classification/transformers_rcnn.yml')
