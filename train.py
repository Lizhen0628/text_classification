import argparse
import collections
import torch
import numpy as np
# import data_loader.data_loaders as module_data
from data_loader import weibo_data_process
from base import base_data_loader
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from trainer import Trainer



# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)

def main(config):
    logger = config.get_logger('train')

    # setup data_set, data_loader instances
    dataset = config.init_obj('dataset',weibo_data_process)
    data_loader = config.init_obj('data_loader',weibo_data_process,dataset=dataset,device=torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    # build model architecture, then print to console
    model = config.init_obj('model_arch', module_arch,vocab_size=len(data_loader.vocab),pad_idx=data_loader.pad_idx)
    logger.info(model)

    if data_loader.use_pretrained_word_embedding:
        model.embedding.weight.data.copy_(data_loader.vocab.vectors)
    # 把unknown 和 pad 向量设置为零
    model.embedding.weight.data[data_loader.unk_idx] = torch.zeros(model.embedding_dim)
    model.embedding.weight.data[data_loader.pad_idx] = torch.zeros(model.embedding_dim)
    # get function handles of loss and metrics
    criterion = getattr(module_loss, config['loss'])
    metrics = [getattr(module_metric, met) for met in config['metrics']]

    # build optimizer, learning rate scheduler. delete every lines containing lr_scheduler for disabling scheduler
    trainable_params = filter(lambda p: p.requires_grad, model.parameters())
    optimizer = config.init_obj('optimizer', torch.optim, trainable_params)

    lr_scheduler = config.init_obj('lr_scheduler', torch.optim.lr_scheduler, optimizer)

    trainer = Trainer(model, criterion, metrics, optimizer,
                      config=config,
                      data_loader=data_loader,
                      lr_scheduler=lr_scheduler)

    trainer.train()


if __name__ == '__main__':
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default='configs/rnn_config.json', type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=None, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_loader;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    main(config)
