'''
Copyright (C) 2010-2021 Alibaba Group Holding Limited.
'''


import os, sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import uuid

import PlainNet
from PlainNet import _get_right_parentheses_index_
from PlainNet.super_blocks import PlainNetSuperBlockClass
from torch import nn
import global_utils
import numpy as np


class SuperResK1KXK1(PlainNetSuperBlockClass):
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None ,sub_layers=None, kernel_size=None,
                 no_create=False, no_reslink=False, no_BN=False, use_se=False, use_sa=False, **kwargs):
        super(SuperResK1KXK1, self).__init__(**kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.bottleneck_channels = bottleneck_channels
        self.sub_layers = sub_layers
        self.kernel_size = kernel_size
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.index_pos = []
        if use_sa is True and use_se is True:
            assert "only choose one block please set again"

        self.use_se = use_se
        self.use_sa = use_sa
        if self.use_se:
            print('---debug use_se in ' + str(self))
        if self.use_sa:
            print('---debug use_sa in ' + str(self))

        full_str = ''
        last_channels = in_channels
        current_stride = stride
        # index_pos format: (i,j) i: index of layer, j index of inner block in one layer => for compute width (custom by JinniPi)
        i = 0
        for j in range(self.sub_layers):
            i = i + 1
            inner_str = ''
            tmp_index_pos = []

            # first bl-block with reslink
            inner_str += 'ConvKX({},{},{},{})'.format(last_channels, self.bottleneck_channels, 1, 1)
            tmp_index_pos += [i]
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.bottleneck_channels)
                tmp_index_pos += [i]
            inner_str += 'RELU({})'.format(self.bottleneck_channels)
            tmp_index_pos += [i]

            inner_str += 'ConvKX({},{},{},{})'.format(self.bottleneck_channels, self.bottleneck_channels,
                                                      self.kernel_size, current_stride)
            tmp_index_pos += [i]
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.bottleneck_channels)
                tmp_index_pos += [i]
            inner_str += 'RELU({})'.format(self.bottleneck_channels)
            tmp_index_pos += [i]
            if self.use_se:
                inner_str += 'SE({})'.format(bottleneck_channels)
                tmp_index_pos += [i]
            elif self.use_sa:
                print("bottleneck_channels", bottleneck_channels)
                if bottleneck_channels >= 32 and ((bottleneck_channels / 8) // 2 == 0):
                    groups = 8
                else:
                    groups = 4
                inner_str += 'SA({},{})'.format(bottleneck_channels, groups)
                tmp_index_pos += [i]

            inner_str += 'ConvKX({},{},{},{})'.format(self.bottleneck_channels, self.out_channels, 1, 1)
            tmp_index_pos += [i]
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.out_channels)
                tmp_index_pos += [i]

            if not self.no_reslink:
                if i == 0:
                    res_str = 'ResBlockProj({})RELU({})'.format(inner_str, out_channels)
                else:
                    res_str = 'ResBlock({})RELU({})'.format(inner_str, out_channels)
                self.index_pos += [i, i]
            else:
                res_str = '{}RELU({})'.format(inner_str, out_channels)
                tmp_index_pos += [i]
                self.index_pos += tmp_index_pos

            full_str += res_str

            # second bl-block with reslink
            i = i + 1
            tmp_index_pos = []
            inner_str = ''
            inner_str += 'ConvKX({},{},{},{})'.format(self.out_channels, self.bottleneck_channels, 1, 1)
            tmp_index_pos += [i]
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.bottleneck_channels)
                tmp_index_pos += [i]
            inner_str += 'RELU({})'.format(self.bottleneck_channels)
            tmp_index_pos += [i]

            inner_str += 'ConvKX({},{},{},{})'.format(self.bottleneck_channels, self.bottleneck_channels,
                                                      self.kernel_size, 1)
            tmp_index_pos += [i]
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.bottleneck_channels)
                tmp_index_pos += [i]
            inner_str += 'RELU({})'.format(self.bottleneck_channels)
            tmp_index_pos += [i]
            if self.use_se:
                inner_str += 'SE({})'.format(bottleneck_channels)
                tmp_index_pos += [i]
            elif self.use_sa:
                if bottleneck_channels >= 32 and ((bottleneck_channels / 8) // 2 == 0):
                    groups = 8
                else:
                    groups = 4
                inner_str += 'SA({},{})'.format(bottleneck_channels, groups)
                tmp_index_pos += [i]

            inner_str += 'ConvKX({},{},{},{})'.format(self.bottleneck_channels, self.out_channels, 1, 1)
            tmp_index_pos += [i]
            if not self.no_BN:
                inner_str += 'BN({})'.format(self.out_channels)
                tmp_index_pos += [i]

            if not self.no_reslink:
                res_str = 'ResBlock({})RELU({})'.format(inner_str, out_channels)
                self.index_pos += [i, i]
            else:
                res_str = '{}RELU({})'.format(inner_str, out_channels)
                tmp_index_pos += [i]
                self.index_pos += tmp_index_pos

            full_str += res_str

            last_channels = out_channels
            current_stride = 1
        pass

        self.block_list = PlainNet.create_netblock_list_from_str(full_str, no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, **kwargs)
        if not no_create:
            self.module_list = nn.ModuleList(self.block_list)
        else:
            self.module_list = None

    def __str__(self):
        return type(self).__name__ + '({},{},{},{},{})'.format(self.in_channels, self.out_channels,
                                                                self.stride, self.bottleneck_channels, self.sub_layers)

    def get_block_num(self):
        return self.sub_layers*2

    def __repr__(self):
        return type(self).__name__ + '({}|in={},out={},stride={},btl_channels={},sub_layers={},kernel_size={})'.format(
            self.block_name, self.in_channels, self.out_channels, self.stride, self.bottleneck_channels, self.sub_layers, self.kernel_size
        )

    def encode_structure(self):
        return [self.out_channels, self.sub_layers, self.bottleneck_channels]

    def split(self, split_layer_threshold):
        if self.sub_layers >= split_layer_threshold:
            new_sublayers_1 = split_layer_threshold // 2
            new_sublayers_2 = self.sub_layers - new_sublayers_1
            new_block_str1 = type(self).__name__ + '({},{},{},{},{})'.format(self.in_channels, self.out_channels,
                                                                self.stride, self.bottleneck_channels, new_sublayers_1)
            new_block_str2 = type(self).__name__ + '({},{},{},{},{})'.format(self.out_channels, self.out_channels,
                                                                             1, self.bottleneck_channels,
                                                                             new_sublayers_2)
            return new_block_str1 + new_block_str2
        else:
            return str(self)

    def structure_scale(self, scale=1.0, channel_scale=None, sub_layer_scale=None):
        if channel_scale is None:
            channel_scale = scale
        if sub_layer_scale is None:
            sub_layer_scale = scale

        new_out_channels = global_utils.smart_round(self.out_channels * channel_scale)
        new_bottleneck_channels = global_utils.smart_round(self.bottleneck_channels * channel_scale)
        new_sub_layers = max(1, round(self.sub_layers * sub_layer_scale))

        return type(self).__name__ + '({},{},{},{},{})'.format(self.in_channels, new_out_channels,
                                                               self.stride, new_bottleneck_channels, new_sub_layers)

    def get_deepmad_forward(self):
        output = np.zeros(self.sub_layers*2)
        for index, block in enumerate(self.block_list):
            # print(index, block)
            output[self.index_pos[index] - 1] += block.get_deepmad_forward()[0]
        return output

    def get_width(self):
        output = np.ones(self.sub_layers*2)
        # tmp_list = []
        for index, block in enumerate(self.block_list):
            # print(index, block)
            # tmp_list.append((block, block.get_width()[0]))
            output[self.index_pos[index] - 1] *= block.get_width()[0]
        # print(tmp_list)
        return output


    @classmethod
    def create_from_str(cls, s, **kwargs):
        assert cls.is_instance_from_str(s)
        idx = _get_right_parentheses_index_(s)
        assert idx is not None
        param_str = s[len(cls.__name__ + '('):idx]

        # find block_name
        tmp_idx = param_str.find('|')
        if tmp_idx < 0:
            tmp_block_name = 'uuid{}'.format(uuid.uuid4().hex)
        else:
            tmp_block_name = param_str[0:tmp_idx]
            param_str = param_str[tmp_idx + 1:]

        param_str_split = param_str.split(',')
        in_channels = int(param_str_split[0])
        out_channels = int(param_str_split[1])
        stride = int(param_str_split[2])
        bottleneck_channels = int(param_str_split[3])
        sub_layers = int(param_str_split[4])
        return cls(in_channels=in_channels, out_channels=out_channels, stride=stride,
                   bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                   block_name=tmp_block_name, **kwargs),s[idx + 1:]


class SuperResK1K3K1(SuperResK1KXK1):
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None, sub_layers=None, no_create=False, **kwargs):
        super(SuperResK1K3K1, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                                           kernel_size=3,
                                           no_create=no_create, **kwargs)

class SuperResK1K5K1(SuperResK1KXK1):
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None, sub_layers=None, no_create=False, **kwargs):
        super(SuperResK1K5K1, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                                           kernel_size=5,
                                           no_create=no_create, **kwargs)


class SuperResK1K7K1(SuperResK1KXK1):
    def __init__(self, in_channels=None, out_channels=None, stride=None, bottleneck_channels=None, sub_layers=None, no_create=False, **kwargs):
        super(SuperResK1K7K1, self).__init__(in_channels=in_channels, out_channels=out_channels, stride=stride,
                                           bottleneck_channels=bottleneck_channels, sub_layers=sub_layers,
                                           kernel_size=7,
                                           no_create=no_create, **kwargs)


def register_netblocks_dict(netblocks_dict: dict):
    this_py_file_netblocks_dict = {
        'SuperResK1K3K1': SuperResK1K3K1,
        'SuperResK1K5K1': SuperResK1K5K1,
        'SuperResK1K7K1': SuperResK1K7K1,
    }
    netblocks_dict.update(this_py_file_netblocks_dict)
    return netblocks_dict

if __name__ == '__main__':

    model = SuperResK1K3K1(in_channels=88, out_channels=120, stride=1, bottleneck_channels=16, sub_layers=1,
                            no_reslink=False)

    out = model.get_width()
    print(out)
    deepmad = model.get_deepmad_forward()
    print(deepmad)
    # print(model.index_pos)
    # print(model.get_model_size())
    print((model.index_pos))
    print(len(model.block_list))
    for block in model.block_list:
        print(block)
