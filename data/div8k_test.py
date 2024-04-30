import os

from data import common
from data import srdata

import numpy as np
import scipy.misc as misc

import torch
import torch.utils.data as data

class Div8k_Test(srdata.SRData):
    def __init__(self, args, name ='', train=True):
        super(Div8k_Test, self).__init__(args, name, train, benchmark=True)

    def _scan(self):
        list_hr = []
        list_lr = [[] for _ in self.scale] 

        for entry in os.scandir(self.dir_hr):
            filename, file_ext = os.path.splitext(entry.name)
            if file_ext == self.ext:
                list_hr.append(os.path.join(self.dir_hr, filename + self.ext))
                for si, s in enumerate(self.scale):
                    list_lr[si].append(os.path.join(
                        self.dir_lr,
                        #'X{}/{}x{}{}'.format(s, filename, s, self.ext)
                        '{}{}'.format(filename, self.ext)
                    ))


        return list_hr, list_lr

    def _set_filesystem(self, dir_data):
        self.apath = os.path.join(dir_data, 'benchmark', self.args.data_test[0])
        #self.apath = os.path.join(dir_data, 'DIV2K', self.name)

        #self.dir_hr = os.path.join(self.apath, 'HR')
        self.dir_hr = os.path.join(self.apath, 'HR', 'X{}'.format(self.scale[0]))
        self.dir_lr = os.path.join(self.apath, 'LR', 'X{}'.format(self.scale[0]))
        self.ext = '.png' 
