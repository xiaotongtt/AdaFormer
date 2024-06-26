import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

# 返回测试图像HR、LR的文件名list
class Benchmark(srdata.SRData):
    def __init__(self, args, name ='', train=True):
        super(Benchmark, self).__init__(args, name, train, benchmark=True)

        #super(Benchmark, self).__init__(args, train)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale]  #不同放大系数

        for entry in os.scandir(self.dir_hr):
            filename, file_ext = os.path.splitext(entry.name)
            if file_ext == self.ext:
                list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
                for si, s in enumerate(self.scale):
                    list_lr[si].append(os.path.join(
                        self.dir_lr,
                        #'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                        'x{}/{}{}'.format(s, filename, self.ext)
                    ))

        # list_hr.sort()
        # for l in list_lr:
        #     l.sort()

        return list_hr, list_lr

    # test的 HR、LR图像路径
    def _set_filesystem(self, dir_data):
        #self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test)
        self.apath = os.path.join(dir_data, 'benchmark', self.name)

        #self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_hr = os.path.join(self.apath, 'HR', 'x4')
        self.dir_lr = os.path.join(self.apath, 'LR_bicubic')
        self.ext = '.png' # 后缀名
