import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
import einops

def window_partition(x, window_size):
    B, H, W, C = x.shape
    x = x.view(B, H // window_size[0], window_size[0], W // window_size[1], window_size[1], C)
    windows = (
        x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, window_size[0]*window_size[1], C)
    )
    return windows


def window_reverse(windows, window_size, H, W):
    nwB, N, C = windows.shape
    windows = windows.view(-1, window_size[0], window_size[1], C)
    B = int(nwB / (H * W / window_size[0] / window_size[1]))
    x = windows.view(
        B, H // window_size[0], W // window_size[1], window_size[0], window_size[1], -1
    )
    x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
    return x


def pad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # center-pad the feature on H and W axes
        img_mask = torch.zeros((1, h+pad_h, w+pad_w, 1))  # 1 H W 1
        h_slices = (
            slice(0, pad_h//2),
            slice(pad_h//2, h+pad_h//2),
            slice(h+pad_h//2, None),
        )
        w_slices = (
            slice(0, pad_w//2),
            slice(pad_w//2, w+pad_w//2),
            slice(w+pad_w//2, None),
        )
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = window_partition(
            img_mask, window_size
        )  # nW, window_size*window_size, 1
        mask_windows = mask_windows.squeeze(-1)
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(
            attn_mask != 0, float(-100.0)
        ).masked_fill(attn_mask == 0, float(0.0))
        return nn.functional.pad(
            x,
            (0, 0, pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2),
        ), attn_mask
    return x, None


def depad_if_needed(x, size, window_size):
    n, h, w, c = size
    pad_h = math.ceil(h / window_size[0]) * window_size[0] - h
    pad_w = math.ceil(w / window_size[1]) * window_size[1] - w
    if pad_h > 0 or pad_w > 0:  # remove the center-padding on feature
        return x[:, pad_h // 2 : pad_h // 2 + h, pad_w // 2 : pad_w // 2 + w, :].contiguous()
    return x


class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, bias=True, groups=dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        x = x.transpose(1, 2).reshape(B, C, H, W)
        x = self.dwconv(x)
        x = x.reshape(B, C, -1).transpose(1, 2)

        return x


class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.dwconv = DWConv(hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)
        self.relu = nn.ReLU(inplace=True)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = self.dwconv(x, H, W)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class AttentionToMotion(nn.Module):
    def __init__(self, dim, window_size, patch_size=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size * self.patch_size

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.mlp = nn.Sequential(
                        nn.Linear(num_heads, num_heads // 2),
                        nn.GELU(),
                        nn.Linear(num_heads // 2, 1),)
        self.apply(self._init_weights)
        self._register_relative_coord_()        

    def _register_relative_coord_(self):
        relative_coord = torch.zeros(1, 1, 2, self.window_size**2, self.window_size**2)
        for y in range(self.window_size):
            for x in range(self.window_size):
                xx, yy = self._get_relative_coord_(x, y)
                relative_coord[:, :, 0, y*self.window_size + x, :] = xx.flatten().unsqueeze(0)
                relative_coord[:, :, 1, y*self.window_size + x, :] = yy.flatten().unsqueeze(0)
        self.register_buffer("relative_coord", relative_coord)

    def _get_relative_coord_(self, x, y, window_size=None):
        if window_size is None:
            window_size = self.window_size
        vx = torch.linspace(-x, window_size-(x+1), steps=window_size)
        vy = torch.linspace(-y, window_size-(y+1), steps=window_size)
        xx, yy = torch.meshgrid(vx, vy, indexing='xy')
        return xx, yy
    
    def _set_window_size_(self, window_size):
        self.window_size = window_size * self.patch_size
        self._register_relative_coord_()
        self.relative_coord = self.relative_coord.to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x1, x2, H, W, mask=None):
        B, N, C = x1.shape
        q = self.q(x1).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        kv = self.kv(x2).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]    
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0] # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        attn = attn.unsqueeze(2) 
        motion = torch.sum(attn * self.relative_coord, dim=-1)
        motion = einops.rearrange(motion, 'B C N L -> (N B) L C') # [B', W*W, 8]
        motion = self.mlp(motion)
        motion = einops.rearrange(motion, '(N B) L C -> B L (N C)', N=2)
        
        return x, motion


class ATMFormer(nn.Module):
    def __init__(self, dim, window_size=7, shift_size=0, patch_size=1, num_heads=8, mlp_ratio=4., bidirectional=True, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        self.window_size = window_size
        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)
        self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = AttentionToMotion(
            dim,
            window_size,
            patch_size=patch_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def _set_window_size_(self, window_size, shift_size=0):
        self.window_size = window_size
        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)

        self.attn._set_window_size_(self.window_size[0])

    def forward(self, x, H, W, B):
        """
            x: feature tensor of shape [2*B, H*W, C], where frame1 and frame2 is stacked in dim 0
            H,W,B: height, width, batch of tensor
        return
            x: window attention enhanced feature tensor of shape [2*B, H*W, C]
            x_motion: bidirectional flow, tensor of shape [2*B, H*W, 2]
        """
        x_pad, mask = pad_if_needed(x, x.size(), self.window_size)

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            
            if hasattr(self, 'HW') and self.HW.item() == H_p * W_p: 
                shift_mask = self.attn_mask
            else:
                shift_mask = torch.zeros((1, H_p, W_p, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size[0]),
                            slice(-self.window_size[0], -self.shift_size[0]),
                            slice(-self.shift_size[0], None))
                w_slices = (slice(0, -self.window_size[1]),
                            slice(-self.window_size[1], -self.shift_size[1]),
                            slice(-self.shift_size[1], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        shift_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(shift_mask, self.window_size).squeeze(-1)  
                shift_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                shift_mask = shift_mask.masked_fill(shift_mask != 0, 
                                float(-100.0)).masked_fill(shift_mask == 0, 
                                float(0.0))
                                
                if mask is not None:
                    shift_mask = shift_mask.masked_fill(mask != 0, 
                                float(-100.0))
                self.register_buffer("attn_mask", shift_mask)
                self.register_buffer("HW", torch.Tensor([H_p*W_p]))
        else: 
            shift_mask = mask
        
        if shift_mask is not None:
            shift_mask = shift_mask.to(x_pad.device)
            
        _, Hw, Ww, C = x_pad.shape
        x_win = window_partition(x_pad, self.window_size)

        nwB = x_win.shape[0]
        x_norm = self.norm1(x_win)

        x_reverse = torch.cat([x_norm[nwB//2:], x_norm[:nwB//2]])
        x_appearence, x_motion = self.attn(x_norm, x_reverse, H, W, shift_mask)
        x_norm = x_norm + self.drop_path(x_appearence)

        x_back = x_norm
        x_back_win = window_reverse(x_back, self.window_size, Hw, Ww)
        x_motion = window_reverse(x_motion, self.window_size, Hw, Ww)
        
        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))
            x_motion = torch.roll(x_motion, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, x.size(), self.window_size).view(2*B, H * W, -1)
        x_motion = depad_if_needed(x_motion, torch.Size([2*B, H, W, 2]), self.window_size).view(2*B, H * W, -1)
            
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x, x_motion
    

class WindowAttention(nn.Module):
    def __init__(self, dim, window_size, patch_size=1, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.dim = dim
        self.patch_size = patch_size
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5
        self.window_size = window_size * self.patch_size

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x, H, W, mask=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            nW = mask.shape[0] # mask: nW, N, N
            attn = attn.view(B // nW, nW, self.num_heads, N, N) + mask.unsqueeze(
                1
            ).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
            attn = attn.softmax(dim=-1)
        else:
            attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class RefineBottleneck(nn.Module):
    def __init__(self, dim, window_size=8, shift_size=0, patch_size=1, num_heads=8, mlp_ratio=4., bidirectional=True, qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm,):
        super().__init__()
        self.window_size = window_size
        if not isinstance(self.window_size, (tuple, list)):
            self.window_size = to_2tuple(window_size)
        self.shift_size = shift_size
        if not isinstance(self.shift_size, (tuple, list)):
            self.shift_size = to_2tuple(shift_size)
        self.bidirectional = bidirectional
        self.norm1 = norm_layer(dim)
        self.attn = WindowAttention(
            dim,
            window_size,
            patch_size=patch_size,
            num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale,
            attn_drop=attn_drop, proj_drop=drop)
        
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
        elif isinstance(m, nn.Conv2d):
            fan_out = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
            fan_out //= m.groups
            m.weight.data.normal_(0, math.sqrt(2.0 / fan_out))
            if m.bias is not None:
                m.bias.data.zero_()

    def forward(self, x):
        """
            x: feature tensor of shape [B, H, W, C]
            H,W,B: height, width, batch of tensor
        return
            x: window attention enhanced feature tensor of shape [B, H*W, C]
        """
        B, H, W, _ = x.size()
        x_pad, mask = pad_if_needed(x, x.size(), self.window_size)

        if self.shift_size[0] or self.shift_size[1]:
            _, H_p, W_p, C = x_pad.shape
            x_pad = torch.roll(x_pad, shifts=(-self.shift_size[0], -self.shift_size[1]), dims=(1, 2))
            
            if hasattr(self, 'HW') and self.HW.item() == H_p * W_p: 
                shift_mask = self.attn_mask
            else:
                shift_mask = torch.zeros((1, H_p, W_p, 1))  # 1 H W 1
                h_slices = (slice(0, -self.window_size[0]),
                            slice(-self.window_size[0], -self.shift_size[0]),
                            slice(-self.shift_size[0], None))
                w_slices = (slice(0, -self.window_size[1]),
                            slice(-self.window_size[1], -self.shift_size[1]),
                            slice(-self.shift_size[1], None))
                cnt = 0
                for h in h_slices:
                    for w in w_slices:
                        shift_mask[:, h, w, :] = cnt
                        cnt += 1

                mask_windows = window_partition(shift_mask, self.window_size).squeeze(-1)  
                shift_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
                shift_mask = shift_mask.masked_fill(shift_mask != 0, 
                                float(-100.0)).masked_fill(shift_mask == 0, 
                                float(0.0))
                                
                if mask is not None:
                    shift_mask = shift_mask.masked_fill(mask != 0, 
                                float(-100.0))
                self.register_buffer("attn_mask", shift_mask)
                self.register_buffer("HW", torch.Tensor([H_p*W_p]))
        else: 
            shift_mask = mask
        
        if shift_mask is not None:
            shift_mask = shift_mask.to(x_pad.device)
            
        _, Hw, Ww, _ = x_pad.shape
        x_win = window_partition(x_pad, self.window_size)
        x_norm = self.norm1(x_win)

        x_appearence = self.attn(x_norm, H, W, shift_mask)
        x_norm = x_norm + self.drop_path(x_appearence)

        x_back = x_norm
        x_back_win = window_reverse(x_back, self.window_size, Hw, Ww)
        
        if self.shift_size[0] or self.shift_size[1]:
            x_back_win = torch.roll(x_back_win, shifts=(self.shift_size[0], self.shift_size[1]), dims=(1, 2))

        x = depad_if_needed(x_back_win, x.size(), self.window_size).view(B, H * W, -1) 
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x





if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    # fa = AttentionToMotion(dim=8, window_size=5).to(device)

    # x = 3
    # y = 2
    # print(fa.relative_coord_x[0,0, y*7+x])
    # print(fa.relative_coord_y[0,0, y*7+x])

    feat_chan = 128
    window_size = 7
    num_heads = 8
    B = 24
    H = 32
    W = 32

    x1 = torch.rand(B, H*W, feat_chan).to(device)
    x2 = torch.rand(B, H*W, feat_chan).to(device)
    x3 = torch.cat([x1,x2], dim=0)

    # k = fa.forward(x1,x2,5,5)

    mf1 = ATMFormer(dim=feat_chan, num_heads=num_heads, window_size=window_size, shift_size=0).to(device)
    x, x_motion1 = mf1.forward(x3, H, W, B)

    print(x.size())
    print(x_motion1.size())    

    mf2 = ATMFormer(dim=feat_chan, num_heads=num_heads, window_size=window_size, shift_size=window_size // 2).to(device)
    x, x_motion2= mf2.forward(x, H, W, B)

    print(x.size())
    print(x_motion2.size())    