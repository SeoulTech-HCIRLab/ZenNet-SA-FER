
import os, sys
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import torch, argparse
from torch import nn
import torch.nn.functional as F
import PlainNet
from PlainNet import parse_cmd_options, _create_netblock_list_from_str_, basic_blocks, super_blocks, SuperResK1KXK1, SuperResIDWEXKX, SuperResKXKX
from PlainNet.lightweight_att import WIRW1
# import random
# random.seed(0)
def parse_cmd_options(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--no_BN', action='store_true')
    parser.add_argument('--no_reslink', action='store_true')
    parser.add_argument('--use_se', action='store_true')
    parser.add_argument('--use_sa', action='store_true')
    parser.add_argument('--use_lw_att', action='store_true')
    module_opt, _ = parser.parse_known_args(argv)
    return module_opt

class Flatten(nn.Module):
    def forward(self, input):
        return input.view(input.size(0), -1)


class MasterNet(PlainNet.PlainNet):
    def __init__(self, argv=None, opt=None, num_classes=None, plainnet_struct=None, no_create=False,
                 no_reslink=None, no_BN=None, use_se=None, use_sa=None, use_lw_att=None):

        if argv is not None:
            module_opt = parse_cmd_options(argv)
        else:
            module_opt = None

        if no_BN is None:
            if module_opt is not None:
                no_BN = module_opt.no_BN
            else:
                no_BN = False

        if no_reslink is None:
            if module_opt is not None:
                no_reslink = module_opt.no_reslink
            else:
                no_reslink = False

        if use_se is None:
            if module_opt is not None:
                use_se = module_opt.use_se
            else:
                use_se = False

        if use_sa is None:
            if module_opt is not None:
                use_sa = module_opt.use_sa
            else:
                use_sa = False

        if use_lw_att is None:
            if module_opt is not None:
                use_lw_att = module_opt.use_lw_att
            else:
                use_lw_att = False


        super().__init__(argv=argv, opt=opt, num_classes=num_classes, plainnet_struct=plainnet_struct,
                                       no_create=no_create, no_reslink=no_reslink, no_BN=no_BN, use_se=use_se, use_sa=use_sa)
        self.last_channels = self.block_list[-1].out_channels
        self.fc_linear = basic_blocks.Linear(in_channels=self.last_channels, out_channels=self.num_classes, no_create=no_create)
        # self.lw_att = sa_layer(n_feats=2048, groups=2048//128)
        # print("use_lw_att: ", use_lw_att)
        if use_lw_att:
            print("debug: You are using lw-att")
            self.lw_att = WIRW1(n_feats=self.last_channels, groups_sa=8)
        self.no_create = no_create
        self.no_reslink = no_reslink
        self.no_BN = no_BN
        self.use_se = use_se
        self.use_sa = use_sa
        self.use_lw_att = use_lw_att
        # bn eps
        for layer in self.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eps = 1e-3


    def extract_stage_features_and_logit(self, x, target_downsample_ratio=None):
        # print("Debug in feature extractor of student network")
        stage_features_list = []
        image_size = x.shape[2]
        output = x

        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)
            dowsample_ratio = round(image_size / output.shape[2])
            # print("type of block: ", type(the_block))
            # print("dowsample_ratio: ", (block_id, dowsample_ratio))
            # print("target", target_downsample_ratio)
            if dowsample_ratio == target_downsample_ratio:
                stage_features_list.append(output)
                target_downsample_ratio *= 2
                # print("feature add to list")
            pass
        pass

        output = F.adaptive_avg_pool2d(output, output_size=1)

        output = torch.flatten(output, 1)
        logit = self.fc_linear(output)
        return stage_features_list, logit

    def forward(self, x):
        output = x
        for block_id, the_block in enumerate(self.block_list):
            output = the_block(output)

        output = F.adaptive_avg_pool2d(output, output_size=1)
        if self.use_lw_att:
            output = self.lw_att(output)

        output = torch.flatten(output, 1)

        output = self.fc_linear(output)

        return output

    def forward_pre_GAP(self, x):
        output = x
        for the_block in self.block_list:
            output = the_block(output)
        return output

    def get_FLOPs(self, input_resolution):
        the_res = input_resolution
        the_flops = 0
        # print("Block list", self.block_list)
        for the_block in self.block_list:
            # print(the_block)
            the_flops += the_block.get_FLOPs(the_res)
            the_res = the_block.get_output_resolution(the_res)

        the_flops += self.fc_linear.get_FLOPs(the_res)

        return the_flops

    def get_model_size(self):
        the_size = 0
        for the_block in self.block_list:
            the_size += the_block.get_model_size()

        the_size += self.fc_linear.get_model_size()

        return the_size

    def get_num_layers(self):
        num_layers = 0
        for block in self.block_list:
            assert isinstance(block, super_blocks.PlainNetSuperBlockClass)
            num_layers += block.sub_layers
        return num_layers

    def replace_block(self, block_id, new_block):
        self.block_list[block_id] = new_block

        if block_id < len(self.block_list) - 1:
            if self.block_list[block_id + 1].in_channels != new_block.out_channels:
                self.block_list[block_id + 1].set_in_channels(new_block.out_channels)
        else:
            assert block_id == len(self.block_list) - 1
            self.last_channels = self.block_list[-1].out_channels
            if self.fc_linear.in_channels != self.last_channels:
                self.fc_linear.set_in_channels(self.last_channels)

        self.module_list = nn.ModuleList(self.block_list)

    def split(self, split_layer_threshold):
        new_str = ''
        for block in self.block_list:
            new_str += block.split(split_layer_threshold=split_layer_threshold)
        return new_str

    def init_parameters(self):
        print("debug here for init model")
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_normal_(m.weight.data, gain=3.26033)
                print('done for innit in Conv2D', m)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
                    print('done for innit in Conv2D bias', m)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                print('done for innit in BatchNorm2d, groupNorm bias', m)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 3.26033 * np.sqrt(2 / (m.weight.shape[0] + m.weight.shape[1])))
                print('done for innit in Linear', m)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.zeros_(m.bias)
            else:
                pass

        for superblock in self.block_list:
            if not isinstance(superblock, super_blocks.PlainNetSuperBlockClass):
                continue
            for block in superblock.block_list:
                if not (isinstance(block, basic_blocks.ResBlock) or isinstance(block, basic_blocks.ResBlockProj)):
                    continue
                # print('---debug set bn weight zero in resblock {}:{}'.format(superblock, block))
                last_bn_block = None
                for inner_resblock in block.block_list:
                    if isinstance(inner_resblock, basic_blocks.BN):
                        last_bn_block = inner_resblock
                    pass
                pass  # end for
                assert last_bn_block is not None
                # print('-------- last_bn_block={}'.format(last_bn_block))
                nn.init.zeros_(last_bn_block.netblock.weight)

    # custom code to compute deepmad score by JinniPi
    def deepmad_forward_pre_GAP(self):
        network_std_list = []
        for idx, the_block in enumerate(self.block_list):
            # print(the_block)
            one_std_list = the_block.get_deepmad_forward().tolist()
            ## type of one std list is array of numpy
            network_std_list += one_std_list
        return network_std_list


    # custom code to compute deepmad score by JinniPi (additional functional)
    def get_stage_info(self, resolution=224):
        stage_idx = []
        stage_channels = []
        stage_block_num = []
        stage_layer_num = []

        stage_feature_map_size = []
        feature_map_size = resolution  # use the input size as initialization

        channel_num = 0
        block_num = 0
        layer_num = 0
        for idx, the_block in enumerate(self.block_list):
            # print("the block", the_block)
            # print("the_block.stride", the_block.stride)

            if the_block.stride == 2 and 0 < idx < len(self.block_list):
                stage_idx.append(idx - 1)
                stage_channels.append(channel_num)
                stage_block_num.append(block_num)
                stage_layer_num.append(layer_num)
                stage_feature_map_size.append(feature_map_size)

            block_num += the_block.get_block_num()
            channel_num = the_block.out_channels
            layer_num += the_block.get_layers()
            # layer_num += the_block.sublayer
            feature_map_size = the_block.get_output_resolution(feature_map_size)

            if idx == len(self.block_list) - 1:
                stage_idx.append(idx)
                stage_channels.append(channel_num)
                stage_block_num.append(block_num)
                stage_layer_num.append(layer_num)
                stage_feature_map_size.append(feature_map_size)

        return stage_idx, stage_block_num, stage_layer_num, stage_channels, stage_feature_map_size

    def get_efficient_score(self, resolution=224):
        stage_idx, stage_block_num, stage_layer_num, stage_channels, stage_feature_map_size = self.get_stage_info(resolution)
        # print(stage_block_num)
        # print(stage_layer_num)
        # print(self.block_list)
        log_width_list = []
        network_block_width_list = []
        for idx, the_block in enumerate(self.block_list):
            one_width = the_block.get_width().tolist()
            # print(one_width)
            network_block_width_list += one_width

        for idx in range(len(stage_layer_num)):

            rho_width = 0.0
            for idx1 in range(stage_block_num[idx]):
                rho_width += np.log(network_block_width_list[idx1])
            log_width_list.append(rho_width)

        log_width_arr = np.array(log_width_list)
        depth_arr = np.array(stage_layer_num)
        effective_score_arr = np.exp(np.log(depth_arr) - log_width_arr / depth_arr)
        return float(np.max(effective_score_arr))

    def get_depth_penalty(self, depth_penalty_ratio):
        if depth_penalty_ratio != 0:
            depth_list_every_block = []
            for block in self.block_list:
                if isinstance(block, (
                SuperResKXKX.SuperResKXKX, SuperResK1KXK1.SuperResK1KXK1, SuperResIDWEXKX.SuperResIDWEXKX)):
                    depth_list_every_block.append(block.get_block_num())

            if isinstance(self.block_list[1], SuperResK1KXK1.SuperResK1KXK1):
                depth_list_every_block = depth_list_every_block[1:]
                # assert len(depth_list_every_block) == 4
            depth_list_every_block = np.array(depth_list_every_block)

            # remove the first block because the first block needs <= 3
            if isinstance(self.block_list[1], (SuperResK1KXK1.SuperResK1KXK1, SuperResKXKX.SuperResKXKX)):
                depth_list_every_block = depth_list_every_block[1:]
            # print("depth_list_every_block after check", depth_list_every_block)
            #
            # print(depth_list_every_block)
            depth_uneven_score = np.exp(np.std(depth_list_every_block))

            # print("depth_uneven_score", depth_uneven_score)
            depth_penalty = depth_uneven_score * depth_penalty_ratio
            return depth_penalty
        else:
            return 0
