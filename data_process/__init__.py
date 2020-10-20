# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 5:08 下午
# @Author  : jeffery
# @FileName: __init__.py.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
# import data_process.weibo_data_process as module_data_process  # weibo
# import data_process.cnews_data_process as module_data_process    # cnews
import data_process.medical_question_data_process as module_data_process # medical question
from torch.utils.data import dataloader as module_dataloader


def makeDataLoader(config):
    # setup data_set, data_process instances
    train_set = config.init_obj('train_set', module_data_process)
    valid_set = config.init_obj('valid_set', module_data_process)
    # test_set = config.init_obj('test_set', module_data_process)

    # print('train num:{}\t valid num:{}\t test num:{}'.format(len(train_set),len(valid_set),len(test_set)))
    train_dataloader = module_dataloader.DataLoader(train_set, batch_size=train_set.batch_size,
                                                    num_workers=train_set.num_workers, collate_fn=train_set.collate_fn)
    valid_dataloader = module_dataloader.DataLoader(valid_set, batch_size=valid_set.batch_size,
                                                    num_workers=valid_set.num_workers, collate_fn=valid_set.collate_fn)
    # test_dataloader = module_dataloader.DataLoader(test_set, batch_size=test_set.batch_size,
    #                                                num_workers=test_set.num_workers, collate_fn=test_set.collate_fn)

    return train_dataloader, valid_dataloader,None

