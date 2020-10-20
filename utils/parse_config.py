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

        """
        method description:
            class to parse configuration json file. Handles hyperparameters for training, initializations of modules, checkpoint saving

        Args:
            config (dict): project parameters
            resume (str): path to the checkpoint being loaded.
            run_id (str): Unique Identifier for training processes. Used to save checkpoints and training log. Timestamp is being used as default
        Returns:
            return_v (rtype): rtype description

        :Author:  jeffery
        :Create:  2020/8/2 10:00 下午
        """
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
    def from_args(cls, cur_config):
        """
        method description:
            Initialize this class from some cli arguments. Used in train, test.

        Args:
            cur_config (dict): project config parameters read

        Returns:
            cls (ConfigParser): Handles hyperparameters for training, initializations of modules, checkpoint saving

        :Author:  jeffery
        :Create:  2020/8/2 8:56 下午
        """
        if cur_config['visual_device'] is not None:
            os.environ["CUDA_VISIBLE_DEVICES"] = cur_config['visual_device']

        if cur_config['resume_path'] is not None:
            # load saved model config
            resume = Path(cur_config['resume_path'])
            cfg_fname = resume.parent / 'config.yml'
        else:
            resume = None
            cfg_fname = Path(cur_config['config_file_name'])

        config = read_yaml(cfg_fname)

        config['device_id'] = cur_config['device_id']

        if resume:
            # update new config for fine-tuning
            config.update(read_yaml(cur_config['config_file_name']))

        return cls(config, resume)

    def init_obj(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        instance initialized with corresponding arguments given.

        `object = config.init_obj('name', module, a, b=1)`
        is equivalent to
        `object = module.name(a, b=1)`
        """
        module_name = self[name]['type']
        module_args = dict(self[name]['args'])
        assert all([k not in module_args for k in kwargs]), 'Overwriting kwargs given in config file is not allowed'
        module_args.update(kwargs)
        return getattr(module, module_name)(*args, **module_args)

    def init_ftn(self, name, module, *args, **kwargs):
        """
        Finds a function handle with the name given as 'type' in config, and returns the
        function with given arguments fixed with functools.partial.

        `function = config.init_ftn('name', module, a, b=1)`
        is equivalent to
        `function = lambda *args, **kwargs: module.name(a, *args, b=1, **kwargs)`.
        """
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