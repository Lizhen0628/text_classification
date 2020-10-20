# -*- coding: utf-8 -*-
# @Time    : 2020/8/20 4:54 下午
# @Author  : jeffery
# @FileName: project_utils.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:


import yaml
import json
from pathlib import Path
from collections import OrderedDict
import sys

def ensure_dir(dirname):
    dirname = Path(dirname)
    if not dirname.is_dir():
        dirname.mkdir(parents=True, exist_ok=False)

def read_yaml(fname):
    fname = Path(fname)
    with fname.open('r',encoding='utf8') as handle:
        return yaml.load(handle,Loader=yaml.Loader)

def read_json(fname):
    fname = Path(fname)
    with fname.open('rt') as handle:
        return json.load(handle, object_hook=OrderedDict)

def write_yaml(content,fname):
    fname = Path(fname)
    with fname.open('w',encoding='utf8') as handle:
        yaml.dump(content,handle)

def write_json(content, fname):
    fname = Path(fname)
    with fname.open('wt') as handle:
        json.dump(content, handle, indent=4, sort_keys=False)