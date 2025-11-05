# -*- coding: utf-8 -*-
# @Time    : 2020/8/27 3:18 下午
# @Author  : jeffery
# @FileName: trainer_utils.py
# @website : http://www.jeffery.ink/
# @github  : https://github.com/jeffery0628
# @Description:

from itertools import repeat
import pandas as pd

def inf_loop(data_loader):
    ''' wrapper function for endless data loader. '''
    for loader in repeat(data_loader):
        yield from loader


class MetricTracker:
    def __init__(self, *keys, writer=None):
        self.writer = writer
        # --- 修改：明确指定 dtype=float ---
        self._data = pd.DataFrame(index=keys, columns=['total', 'counts', 'average'], dtype=float)
        self.reset()

    def reset(self):
        # --- 修改：用 0.0 (float) 填充 ---
        for col in self._data.columns:
            self._data[col].values[:] = 0.0

    def update(self, key, value, n=1):
        # 警告：这里的 'value' 必须是数字 (int 或 float)
        # 绝不能是 timedelta
        if self.writer is not None:
            self.writer.add_scalar(key, value)
            
        self._data.loc[key, 'total'] += value * n
        self._data.loc[key, 'counts'] += n
        
        # 添加一个除零保护
        if self._data.loc[key, 'counts'] > 0:
            self._data.loc[key, 'average'] = self._data.loc[key, 'total'] / self._data.loc[key, 'counts']
        else:
            self._data.loc[key, 'average'] = 0.0

    def avg(self, key):
        return self._data.average[key]

    def result(self):
        return dict(self._data.average)