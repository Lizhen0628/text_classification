# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 4:53 下午
# @Author  : jeffery
# @FileName: parse_config.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:
from pathlib import Path
import logging
from pathlib import Path
from functools import reduce, partial
from operator import getitem
from datetime import datetime
from logger import setup_logging
from utils import read_yaml,write_yaml
import os

class ConfigParser:
    def __init__(self, config, resume=None, run_id=None):
        # load config file
        self._config = config
        self.resume = resume

        # set save_dir where trained model and log will be saved.
        save_dir = Path(self.config['trainer']['save_dir'])

        exper_name = self.config['experiment_name']
        if run_id is None:  # use timestamp as default run-id
            run_id = datetime.now().strftime(r'%m%d_%H%M%S')
        self._save_dir = save_dir / 'models' / exper_name / run_id
        self._log_dir = save_dir / 'log' / exper_name / run_id
        self._diff_dir = save_dir / 'diff' / exper_name / run_id

        # make directory for saving checkpoints and log.
        exist_ok = run_id == ''
        self.save_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.log_dir.mkdir(parents=True, exist_ok=exist_ok)
        self.diff_dir.mkdir(parents=True,exist_ok=exist_ok)

        # save updated config file to the checkpoint dir
        write_yaml(self.config,self.save_dir / 'config.yml')

        # configure logging module
        setup_logging(self.log_dir)
        self.log_levels = {
            0: logging.WARNING,
            1: logging.INFO,
            2: logging.DEBUG
        }

    @classmethod
    def from_args(cls, args_dict: dict):
        visual_device = args_dict.get('device', None)
        if visual_device is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = visual_device

        resume_path = args_dict.get('resume', None)
        config_path = args_dict.get('config', None)


        # 注意，这里如果resume_path不为空，代表需要继续按照上一个checkpoint进行训练；否则读取配置的config文件
        # 因此这里读取的config文件应该是处于saved文件夹下的config，而不是初始的config文件
        if resume_path is not None:
            resume = Path(resume_path)
            cfg_fname = resume.parent / 'config.yml'
        else:
            resume = None
            if config_path is None:
                raise ValueError("No config file provided (--config)")
            cfg_fname = Path(config_path)


        # 用yaml方式读取config文件
        config = read_yaml(cfg_fname) 
        # 用读入的resume_path和device_id更新config对应的键值对内容
        config['resume_path'] = resume_path
        config['device_id'] = visual_device
        # cls返回的是configParser类本身
        return cls(config=config, resume=resume)

    def init_obj(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return partial(getattr(module, module_name), *args, **module_args)

    def __getitem__(self, name):
        """Access items like ordinary dict."""
        return self.config[name]

    def get_logger(self, name, verbosity=2):
        msg_verbosity = 'verbosity option {} is invalid. Valid options are {}.'.format(verbosity,
                                                                                       self.log_levels.keys())
        assert verbosity in self.log_levels, msg_verbosity
        logger = logging.getLogger(name)
        logger.setLevel(self.log_levels[verbosity])
        return logger

    # setting read-only attributes
    @property
    def config(self):
        return self._config

    @property
    def save_dir(self):
        return self._save_dir

    @property
    def log_dir(self):
        return self._log_dir

    @property
    def diff_dir(self):
        return self._diff_dir