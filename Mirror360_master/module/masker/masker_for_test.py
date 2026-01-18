
import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from module.masker.base_masker import BaseMaskerSystem, MaskGenerate
from einops import rearrange
class RandomDynamicMaskSyetem(nn.Module):

    def __init__(self, patch_size=16,num_frames=5,device='cuda'):
            super().__init__()
            self.patch_size = patch_size
            self.num_frames = num_frames
            self.device = device
    def forward(self,mask_frame,mask_ratio):#这个处理方式可以兼用逐帧mask和单帧mask
        B,T,C,H,W=mask_frame.shape
        patched_pic = rearrange(mask_frame, 'b t c (h_p p1) (w_p p2) -> b t (h_p w_p) (c p1 p2)',
                                p1=self.patch_size, p2=self.patch_size
                                )# b t N P
        final_mask = torch.zeros(patched_pic.shape[0],patched_pic.shape[1],patched_pic.shape[2])
        num_to_mask = int(mask_ratio * patched_pic.shape[2])
        
        for b in range(B):
            for t in range(T):
                if num_to_mask == 0:
                    continue
                layer_indices = torch.arange(patched_pic.shape[2], device=self.device)
                perm = torch.randperm(len(layer_indices), device=self.device)
                selected = layer_indices[perm[:num_to_mask]]

                final_mask[b,t,selected] = 1.0
        final_mask = rearrange(final_mask, 'b t (h w) -> b t h w', h=H//self.patch_size, w=W//self.patch_size)
        final_mask = final_mask.to(torch.bool)    
        return final_mask


    
class RandomMaskerSystem(BaseMaskerSystem):
    def __init__(self, patch_size=16,num_frames=5,device='cuda'):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
    def forward(self,mask_frame,mask_ratio):#这个处理方式可以兼用逐帧mask和单帧mask
        B,T,C,H,W=mask_frame.shape
        patched_pic = rearrange(mask_frame, 'b t c (h_p p1) (w_p p2) -> b t (h_p w_p) (c p1 p2)',
                                p1=self.patch_size, p2=self.patch_size
                                )# b t N P
        final_mask = torch.zeros(patched_pic.shape[0],patched_pic.shape[1],patched_pic.shape[2])
        num_to_mask = int(mask_ratio * patched_pic.shape[2])
        
        for b in range(B):

            if num_to_mask == 0:
                continue
            layer_indices = torch.arange(patched_pic.shape[2], device=self.device)
            perm = torch.randperm(len(layer_indices), device=self.device)
            selected = layer_indices[perm[:num_to_mask]]
            for t in range(T):
                final_mask[b,t,selected] = 1.0
        final_mask = rearrange(final_mask, 'b t (h w) -> b t h w', h=H//self.patch_size, w=W//self.patch_size)
        final_mask = final_mask.to(torch.bool)  
        return final_mask
class GridMasker(BaseMaskerSystem):
    def __init__(self,patch_size=8,num_frames=5,device='cuda'):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.device = device
    def forward(self,mask_frame,mask_num):
        B,T,C,H,W=mask_frame.shape
        patched_pic = rearrange(mask_frame, 'b t c (h_p sp1 p1) (w_p sp2 p2) -> b t  (h_p w_p) (sp1 sp2) (c p1 p2)',
                                p1=self.patch_size, p2=self.patch_size,
                                sp1=2,sp2=2
                                )# b t 4 N P
        final_mask = torch.zeros(patched_pic.shape[0],patched_pic.shape[1],patched_pic.shape[2],patched_pic.shape[3])
        mask_num = torch.randperm(3)[:2] 
        for num in mask_num:
            final_mask [:,:,:,num]=1.0
        final_mask = rearrange(final_mask,'b t (h_p w_p) (sp1 sp2) -> b t (h_p sp1) (w_p sp2)',
                               h_p= H//self.patch_size//2,w_p=W//self.patch_size//2,sp1=2,sp2=2)
        final_mask = final_mask.to(torch.bool)
        return final_mask
class RandomGridMasker(BaseMaskerSystem):
    def __init__(self,patch_size=8,num_frames=5,device='cuda'):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.device = device
    def forward(self,mask_frame,mask_num):
        B,T,C,H,W=mask_frame.shape
        mask_num=2
        patched_pic = rearrange(mask_frame, 'b t c (h_p sp1 p1) (w_p sp2 p2) -> b t (h_p w_p) (sp1 sp2)  (c p1 p2)',
                                p1=self.patch_size, p2=self.patch_size,
                                sp1=2,sp2=2
                                )# b t 4 N P
        final_mask = torch.zeros(patched_pic.shape[0],patched_pic.shape[1],patched_pic.shape[2],patched_pic.shape[3])
        for b in range(B):
            for t in range(T):
                for patch_num in range(final_mask.shape[2]):
                    mask_num = torch.randperm(3)[:2] 
                    final_mask[b,t,patch_num,mask_num]=1.0
        
        final_mask = rearrange(final_mask,'b t (h_p w_p) (sp1 sp2) -> b t (h_p sp1) (w_p sp2)',
                               h_p= H//self.patch_size//2,w_p=W//self.patch_size//2,sp1=2,sp2=2)

        final_mask = final_mask.to(torch.bool)
        return final_mask
class FixedMaskerSystem(BaseMaskerSystem):
    def __init__(self,patch_size=8,num_frames=5,device='cuda'):
        super().__init__()
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.device = device
    def forward(self,mask_frame,mask_num):
        B,T,C,H,W=mask_frame.shape
        patched_pic = rearrange(mask_frame, 'b t c (h_p sp1 p1) (w_p sp2 p2) -> b t  (h_p w_p) (sp1 sp2) (c p1 p2)',
                                p1=self.patch_size, p2=self.patch_size,
                                sp1=2,sp2=2
                                )# b t 4 N P
        final_mask = torch.zeros(patched_pic.shape[0],patched_pic.shape[1],patched_pic.shape[2],patched_pic.shape[3])
        final_mask[:,0,:,0] =1.0
        final_mask[:,0,:,1] =1.0
        final_mask[:,1,:,1] =1.0
        final_mask[:,1,:,3] =1.0
        final_mask[:,2,:,2] =1.0
        final_mask[:,2,:,3] =1.0
        final_mask[:,3,:,3] =1.0
        final_mask[:,3,:,0] =1.0
        final_mask[:,4,:,0] =1.0
        final_mask[:,4,:,1] =1.0
        final_mask = rearrange(final_mask,'b t (h_p w_p) (sp1 sp2) -> b t (h_p sp1) (w_p sp2)',
                               h_p= H//self.patch_size//2,w_p=W//self.patch_size//2,sp1=2,sp2=2)

        final_mask = final_mask.to(torch.bool)
        return final_mask

