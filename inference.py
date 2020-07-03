# -*- coding: utf-8 -*-
# @Time    : 2020/7/3 12:45 下午
# @Author  : lizhen
# @FileName: inference.py
# @Description:

import argparse
import collections
import torch
import numpy as np
from tqdm import tqdm
# import data_process.data_loaders as module_data
from data_process import data_process as module_data_process
from torch.utils.data import dataloader as module_dataloader
from base import base_dataset
import model.loss as module_loss
import model.metric as module_metric
import model.model as module_arch
from parse_config import ConfigParser
from torch.functional import F



# fix random seeds for reproducibility
SEED = 123
torch.manual_seed(SEED)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(SEED)


def main(config):
    logger = config.get_logger('test')
    logger = config.get_logger('train')
    device = torch.device('cuda:{}'.format(config.config['device_id']) if config.config['n_gpu'] > 0 else 'cpu')

    # setup data_set, data_process instances
    dataset = config.init_obj('dataset', module_data_process, device=device)
    # setup data_loader instances
    test_dataloader = config.init_obj('data_loader', module_dataloader, dataset=dataset.test_set,
                                      collate_fn=dataset.collate_fn_4_inference)

    # build model architecture, then print to console
    model = config.init_obj('model_arch', module_arch, word_embedding=dataset.word_embedding)
    logger.info(model)

    logger.info('Loading checkpoint: {} ...'.format(config.resume))
    checkpoint = torch.load(config.resume)
    state_dict = checkpoint['state_dict']
    if config['n_gpu'] > 1:
        device_ids = list(map(lambda x: int(x), config.config['device_id'].split(',')))
        model = torch.nn.DataParallel(model, device_ids=device_ids)
    model.load_state_dict(state_dict)

    # prepare model for testing
    model = model.cuda()
    model.eval()
    label_id_map = {'体育': 0, '娱乐': 1, '家居': 2, '房产': 3, '教育': 4, '时尚': 5, '时政': 6, '游戏': 7, '科技': 8, '财经': 9}
    id_map_label = {0:'体育', 1:'娱乐', 2:'家居', 3:'房产', 4:'教育', 5:'时尚', 6:'时政', 7:'游戏', 8:'科技', 9:'财经'}
    # inference
    with torch.no_grad():
        for i, batch_data in enumerate(test_dataloader):
            input_token_ids, _, seq_lens,class_labels,texts = batch_data

            output = model(input_token_ids, _, seq_lens).squeeze(1)
            output = torch.argmax(F.softmax(output),dim=-1)
            output = output.cpu().detach().numpy()
            class_labels = class_labels.cpu().detach().numpy()
            for text,pred,label in zip(texts,output,class_labels):
                print('text:{}\npredict label:{}\ntrue label:{}'.format(text,id_map_label[pred],id_map_label[label]))
                print('--'*50)






def run(config_file, model_path):
    args = argparse.ArgumentParser(description='text classification')
    args.add_argument('-c', '--config', default=config_file, type=str,
                      help='config file path (default: None)')
    args.add_argument('-r', '--resume', default=model_path, type=str,
                      help='path to latest checkpoint (default: None)')
    args.add_argument('-d', '--device', default='0', type=str,
                      help='indices of GPUs to enable (default: all)')

    # custom cli options to modify configuration from default values given in json file.
    CustomArgs = collections.namedtuple('CustomArgs', 'flags type target')
    options = [
        CustomArgs(['--lr', '--learning_rate'], type=float, target='optimizer;args;lr'),
        CustomArgs(['--bs', '--batch_size'], type=int, target='data_process;args;batch_size')
    ]
    config = ConfigParser.from_args(args, options)
    print(config.config['model_arch']['type'].lower())

    main(config)


if __name__ == '__main__':
    run('configs/multi_classification/fast_text_config.json', '/home/lizhen/workspace/saved/fast_text/models/text_binary_classification/0703_124516/model_best.pth')
