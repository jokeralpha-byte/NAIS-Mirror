"""
修改后的 deformable_alignment.py - 完全兼容原始权重
"""
import torch
import torch.nn as nn
from torch.ao.quantization import QuantStub, DeQuantStub

from mmcv.ops import ModulatedDeformConv2d, modulated_deform_conv2d
from mmengine.model import constant_init

from model.inpainter.flow_comp import flow_warp


class QuantizableSecondOrderDeformableAlignment(ModulatedDeformConv2d):
    """
    量化友好的 Second-order deformable alignment
    关键改动：
    1. 继续继承 ModulatedDeformConv2d，保持权重键名不变
    2. 添加 QuantStub/DeQuantStub 控制量化边界
    3. inplace=False 提高量化稳定性
    """
    def __init__(self, *args, **kwargs):
        self.max_residue_magnitude = kwargs.pop('max_residue_magnitude', 10)
        
        super(QuantizableSecondOrderDeformableAlignment, self).__init__(*args, **kwargs)
        
        # Offset 预测网络
        self.conv_offset = nn.Sequential(
            nn.Conv2d(3 * self.out_channels + 4, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),  # 改为 False
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(self.out_channels, self.out_channels, 3, 1, 1),
            nn.LeakyReLU(negative_slope=0.1, inplace=False),
            nn.Conv2d(self.out_channels, 27 * self.deform_groups, 3, 1, 1),
        )
        
        # 量化控制（用于保护 DCN 操作）
        self.quant_input = QuantStub()
        self.dequant_input = DeQuantStub()
        self.quant_output = QuantStub()
        self.dequant_output = DeQuantStub()
        
        self.init_offset()
    
    def init_offset(self):
        constant_init(self.conv_offset[-1], val=0, bias=0)
    
    def forward(self, x, extra_feat, flow_1, flow_2):
        # 拼接输入
        extra_feat = torch.cat([extra_feat, flow_1, flow_2], dim=1)
        
        # Offset 预测（可量化部分）
        out = self.conv_offset(extra_feat)
        o1, o2, mask = torch.chunk(out, 3, dim=1)
        
        # === 关键：DCN 操作需要在浮点精度下执行 ===
        # 反量化所有输入
        o1_fp = self.dequant_input(o1)
        o2_fp = self.dequant_input(o2)
        mask_fp = self.dequant_input(mask)
        flow_1_fp = self.dequant_input(flow_1)
        flow_2_fp = self.dequant_input(flow_2)
        x_fp = self.dequant_input(x)
        
        # Offset 计算（浮点）
        offset = self.max_residue_magnitude * torch.tanh(
            torch.cat((o1_fp, o2_fp), dim=1)
        )
        offset_1, offset_2 = torch.chunk(offset, 2, dim=1)
        
        offset_1 = offset_1 + flow_1_fp.flip(1).repeat(1, offset_1.size(1) // 2, 1, 1)
        offset_2 = offset_2 + flow_2_fp.flip(1).repeat(1, offset_2.size(1) // 2, 1, 1)
        offset = torch.cat([offset_1, offset_2], dim=1)
        
        # Mask
        mask = torch.sigmoid(mask_fp)
        
        # DCN 操作（浮点）
        output = modulated_deform_conv2d(
            x_fp, offset, mask,
            self.weight, self.bias,  # 注意：直接使用 self.weight/bias，与原始模型一致
            self.stride, self.padding,
            self.dilation, self.groups,
            self.deform_groups
        )
        
        # 重新量化输出
        output = self.quant_output(output)
        
        return output


class QuantizableBidirectionalPropagation(nn.Module):
    """量化友好的双向传播模块"""
    
    def __init__(self, channel):
        super(QuantizableBidirectionalPropagation, self).__init__()
        modules = ['backward_', 'forward_']
        self.deform_align = nn.ModuleDict()
        self.backbone = nn.ModuleDict()
        self.channel = channel
        
        for i, module in enumerate(modules):
            # 使用量化友好的 Deformable Alignment
            self.deform_align[module] = QuantizableSecondOrderDeformableAlignment(
                2 * channel, channel, 3, padding=1, deform_groups=16
            )
            
            self.backbone[module] = nn.Sequential(
                nn.Conv2d((2 + i) * channel, channel, 3, 1, 1),
                nn.LeakyReLU(negative_slope=0.1, inplace=False),  # 改为 False
                nn.Conv2d(channel, channel, 3, 1, 1),
            )
        
        self.fusion = nn.Conv2d(2 * channel, channel, 1, 1, 0)
        
        # 量化控制
        self.quant = QuantStub()
        self.dequant = DeQuantStub()
    
    def forward(self, x, flows_backward, flows_forward):
        """
        x shape: [b, t, c, h, w]
        return: [b, t, c, h, w]
        """
        b, t, c, h, w = x.shape
        feats = {}
        feats['spatial'] = [x[:, i, :, :, :] for i in range(0, t)]
        
        for module_name in ['backward_', 'forward_']:
            feats[module_name] = []
            
            frame_idx = range(0, t)
            flow_idx = range(-1, t - 1)
            mapping_idx = list(range(0, len(feats['spatial'])))
            mapping_idx += mapping_idx[::-1]
            
            if 'backward' in module_name:
                frame_idx = frame_idx[::-1]
                flows = flows_backward
            else:
                flows = flows_forward
            
            feat_prop = x.new_zeros(b, self.channel, h, w)
            
            for i, idx in enumerate(frame_idx):
                feat_current = feats['spatial'][mapping_idx[idx]]
                
                if i > 0:
                    flow_n1 = flows[:, flow_idx[i], :, :, :]
                    
                    # === flow_warp 需要浮点精度 ===
                    feat_prop_fp = self.dequant(feat_prop)
                    cond_n1 = flow_warp(feat_prop_fp, flow_n1.permute(0, 2, 3, 1))
                    cond_n1 = self.quant(cond_n1)
                    
                    # 初始化二阶特征
                    feat_n2 = torch.zeros_like(feat_prop)
                    flow_n2 = torch.zeros_like(flow_n1)
                    cond_n2 = torch.zeros_like(cond_n1)
                    
                    if i > 1:
                        feat_n2 = feats[module_name][-2]
                        flow_n2 = flows[:, flow_idx[i - 1], :, :, :]
                        
                        feat_n2_fp = self.dequant(feat_n2)
                        flow_n2 = flow_n1 + flow_warp(flow_n2, flow_n1.permute(0, 2, 3, 1))
                        cond_n2 = flow_warp(feat_n2_fp, flow_n2.permute(0, 2, 3, 1))
                        cond_n2 = self.quant(cond_n2)
                    
                    cond = torch.cat([cond_n1, feat_current, cond_n2], dim=1)
                    feat_prop = torch.cat([feat_prop, feat_n2], dim=1)
                    feat_prop = self.deform_align[module_name](feat_prop, cond, flow_n1, flow_n2)
                
                feat = [feat_current] + [
                    feats[k][idx]
                    for k in feats if k not in ['spatial', module_name]
                ] + [feat_prop]
                
                feat = torch.cat(feat, dim=1)
                feat_prop = feat_prop + self.backbone[module_name](feat)
                feats[module_name].append(feat_prop)
            
            if 'backward' in module_name:
                feats[module_name] = feats[module_name][::-1]
        
        outputs = []
        for i in range(0, t):
            align_feats = [feats[k].pop(0) for k in feats if k != 'spatial']
            align_feats = torch.cat(align_feats, dim=1)
            outputs.append(self.fusion(align_feats))
        
        return torch.stack(outputs, dim=1) + x


# 别名，保持兼容性
SecondOrderDeformableAlignment = QuantizableSecondOrderDeformableAlignment
BidirectionalPropagation = QuantizableBidirectionalPropagation
