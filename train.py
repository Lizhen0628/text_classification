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


#重要！！！不加的话，lr scheduler无法发挥作用
from transformers import get_linear_schedule_with_warmup
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
    from trainer.weibo_trainer import Trainer # weibo
    # from trainer.cnews_trainer import Trainer # cnews
    # from trainer.medical_question_trainer import Trainer

    # 将训练的log保存为logger存储
    logger = config.get_logger('train')
    # 建立train，valid和tetst的dataloader用于批量训练
    train_dataloader, valid_dataloader, test_dataloader = makeDataLoader(config)

    # get model architecture from config
    model = makeModel(config)
    logger.info(model)
    # get loss function from config
    criterion = makeLoss(config)
    # get metrics from config
    metrics = makeMetrics(config)

    # get optimizer from config
    optimizer = makeOptimizer(config, model)

    # get lr_scheduler from config
    epochs = config['trainer']['epochs']
    steps_per_epoch = len(train_dataloader)
    num_training_steps = steps_per_epoch * epochs
    num_warmup_steps = config.config['lr_scheduler']['args']['num_warmup_steps']
    lr_scheduler = get_linear_schedule_with_warmup(
                    optimizer,
                    num_warmup_steps=num_warmup_steps,
                    num_training_steps=num_training_steps
                )
    # instanialize trainer with all args
    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=train_dataloader,
                      valid_data_loader=valid_dataloader,
                      test_data_loader=test_dataloader,
                      lr_scheduler=lr_scheduler)
    trainer.train()


def run(config_fname):
    # 第一步：将config文件的内容和其余两个参数传入dict字典
    args_dict = {
        'config': config_fname,
        'resume': None, # 明确传入 None
        'device': '0'  # 明确传入 None (from_args 会检查这个)
    }
    # 第二步：将configparser进行初始化，将config作为configParser实例对象
    config = ConfigParser.from_args(args_dict)
    main(config)


if __name__ == '__main__':
    run('configs/binary_classification/word_embedding_rnn.yml')
    # run('configs/binary_classification/word_embedding_rnn_attention.yml')
    # run('configs/multi_classification/word_embedding_text_cnn_1d.yml')
    # run('configs/multi_classification/word_embedding_fast_text.yml')
    # run('configs/multi_classification/word_embedding_rnn.yml')
    # run('configs/multi_classification/word_embedding_rcnn.yml')
    # run('configs/multi_classification/word_embedding_rnn_attention.yml')
    # run('configs/multi_classification/word_embedding_dpcnn.yml')

    # run('configs/multi_classification/transformers_pure.yml')
    # run('configs/multi_classification/transformers_cnn.yml')
    # run('configs/multi_classification/transformers_rnn.yml')
    # run('configs/multi_classification/transformers_rcnn.yml')
