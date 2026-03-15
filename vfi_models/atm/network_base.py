import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_
import einops
import math
from .flow_warp import flow_warp
from .attention import ATMFormer
from .attention import RefineBottleneck as SwinTransformer

def upsample_flow(flow, upsample_factor=2, mode='bilinear'):
	if mode == 'nearest':
		up_flow = F.interpolate(flow, scale_factor=upsample_factor,
								mode=mode) * upsample_factor
	else:
		up_flow = F.interpolate(flow, scale_factor=upsample_factor,
								mode=mode, align_corners=True) * upsample_factor
	return up_flow	
	
def conv(in_dim, out_dim, kernel_size=3, stride=1, padding=1, dilation=1):
	return nn.Sequential(
		nn.Conv2d(in_dim, out_dim, kernel_size=kernel_size, stride=stride,
				  padding=padding, dilation=dilation, bias=True),
		nn.PReLU(out_dim)
		)

def deconv(in_dim, out_dim, kernel_size=4, stride=2, padding=1):
	return nn.Sequential(
		nn.ConvTranspose2d(in_dim, out_dim, kernel_size=kernel_size, 
						   stride=stride, padding=padding, bias=True),
		nn.PReLU(out_dim)
		)    

class CrossScaleFeatureFusion(nn.Module):
	def __init__(self, in_dims=[32, 64, 128, 256], fused_dim=None, conv=nn.Conv2d):
		super().__init__()
		
		layers = []
		for i in range(len(in_dims)-1):
			for j in range(2 ** i):
				layers.append(
					conv(in_channels=in_dims[-2-i],
						out_channels=in_dims[-2-i],
						kernel_size=3,
						stride=2**(i+1),
						padding=1+j,
						dilation=1+j,
						bias=True)
					)
		self.layers = nn.ModuleList(layers)
		concat_dim = sum([2**(len(in_dims)-2-i) * in_dims[i] for i in range(len(in_dims)-1)]) + in_dims[-1]
		if fused_dim is None:
			fused_dim = concat_dim
		self.proj = nn.Conv2d(concat_dim, fused_dim, 1, 1)
		self.norm = nn.LayerNorm(fused_dim)
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

	def forward(self, xs):
		ys = []
		k = 0
		for i in range(len(xs)-1):
			for _ in range(2 ** i):
				ys.append(self.layers[k](xs[-2-i]))
				k += 1
		ys.append(xs[-1])
		x = self.proj(torch.cat(ys, dim=1))
		_, _, H, W = x.shape
		x = x.flatten(2).transpose(1, 2)
		x = self.norm(x)
		return x, H, W


class Network(nn.Module):
	def __init__(self, global_motion=True, ensemble_global_motion=False):
		super(Network, self).__init__()	
		self.pyramid_level = 4
		self.hidden_dims = [24, 48, 96, 192]
		assert len(self.hidden_dims) == self.pyramid_level

		self.global_motion = global_motion
		self.ensemble_global_motion = ensemble_global_motion
				
		# pyramid feature extraction
		self.feat_extracts = nn.ModuleList([])
		for i in range(self.pyramid_level):
			if i == 0:
				self.feat_extracts.append(
						nn.Sequential(conv(3, self.hidden_dims[i], kernel_size=3, stride=1, padding=1),
									  conv(self.hidden_dims[i], self.hidden_dims[i], kernel_size=3, stride=1, padding=1))
					)
			else:
				self.feat_extracts.append(
						nn.Sequential(conv(self.hidden_dims[i-1], self.hidden_dims[i], kernel_size=3, stride=2, padding=1),
									  conv(self.hidden_dims[i], self.hidden_dims[i], kernel_size=3, stride=1, padding=1))
					)
		
		# ----- local motion -----
		concat_dim = self.hidden_dims[-1] + self.hidden_dims[-2] + 2*self.hidden_dims[-3]
		fused_dim = concat_dim
		self.cross_scale_feature_fusion = CrossScaleFeatureFusion(in_dims=self.hidden_dims[1:], fused_dim=fused_dim)

		self.local_motion_args = {
			"window_size": 8,
			"num_heads": 8,
			"patch_size": 1,
			"dim": fused_dim,
			"enhance_window": 8,
		}

		# feature enhancement swin transformer
		self.feat_enhance_transformer = nn.ModuleList([
										SwinTransformer(dim=self.local_motion_args["dim"], 
														window_size=self.local_motion_args["enhance_window"], 
														shift_size=0, 
														patch_size=self.local_motion_args["patch_size"], 
														num_heads=self.local_motion_args["num_heads"], ),
										SwinTransformer(dim=self.local_motion_args["dim"], 
														window_size=self.local_motion_args["enhance_window"], 
														shift_size=self.local_motion_args["enhance_window"]//2, 
														patch_size=self.local_motion_args["patch_size"], 
														num_heads=self.local_motion_args["num_heads"], )
									])

		self.local_motion_atmformer = nn.ModuleList([
										ATMFormer(dim=self.local_motion_args["dim"], 
														  window_size=self.local_motion_args["window_size"], 
														  shift_size=0, 
														  patch_size=self.local_motion_args["patch_size"], 
														  num_heads=self.local_motion_args["num_heads"], ),
										ATMFormer(dim=self.local_motion_args["dim"], 
														  window_size=self.local_motion_args["window_size"], 
														  shift_size=self.local_motion_args["window_size"]//2, 
														  patch_size=self.local_motion_args["patch_size"], 
														  num_heads=self.local_motion_args["num_heads"], )
									])
		
		self.fused_dim = fused_dim * 2
		self.motion_out_dim = 5
		motion_mlp_hidden_dim = int(self.fused_dim * 0.75)
		self.local_motion_mlp = nn.Sequential(
								conv(self.fused_dim + self.local_motion_args["num_heads"], motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
								conv(motion_mlp_hidden_dim, motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
								nn.Conv2d(motion_mlp_hidden_dim, self.motion_out_dim, kernel_size=1, stride=1, padding=0)
							)
		
		# ----- global motion -----
		last_feat_dim = self.hidden_dims[-1] + 96
		self.last_feat_extract = nn.Sequential(
									conv(self.hidden_dims[-1], last_feat_dim, kernel_size=3, stride=2, padding=1),
									conv(last_feat_dim, last_feat_dim, kernel_size=3, stride=1, padding=1)
								)
		
		concat_dim = last_feat_dim + self.hidden_dims[-1] + 2*self.hidden_dims[-2]
		self.global_feature_fusion = CrossScaleFeatureFusion(in_dims=[self.hidden_dims[-2], self.hidden_dims[-1], last_feat_dim], fused_dim=concat_dim)

		self.global_motion_args = {
			"window_size": 12,
			"num_heads": 8,
			"patch_size": 1,
			"dim": concat_dim,
		}

		self.global_motion_atmformer = nn.ModuleList([
										ATMFormer(dim=self.global_motion_args["dim"], 
														  window_size=self.global_motion_args["window_size"], 
														  shift_size=0, 
														  patch_size=self.global_motion_args["patch_size"], 
														  num_heads=self.global_motion_args["num_heads"], ),
										ATMFormer(dim=self.global_motion_args["dim"], 
														  window_size=self.global_motion_args["window_size"], 
														  shift_size=self.global_motion_args["window_size"]//2, 
														  patch_size=self.global_motion_args["patch_size"], 
														  num_heads=self.global_motion_args["num_heads"], )
									])
				
		motion_mlp_hidden_dim = 768
		self.global_motion_mlp = nn.Sequential(
								conv(concat_dim*2 + self.global_motion_args["num_heads"], motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
								conv(motion_mlp_hidden_dim, motion_mlp_hidden_dim, kernel_size=3, stride=1, padding=1),
								nn.Conv2d(motion_mlp_hidden_dim, self.motion_out_dim, kernel_size=1, stride=1, padding=0)
							)
				
		self.fused_dim1 = self.fused_dim // 2
		self.fused_dim2 = self.fused_dim // 4
		self.fused_dim3 = self.fused_dim // 8
		self.fused_dims = [self.fused_dim1, self.fused_dim2, self.fused_dim3, 2*self.fused_dim1]
		deconv_args = {'kernel_size':2, 'stride':2, 'padding':0}
		self.upsample_pyramid = nn.ModuleList([
							nn.Sequential(deconv(self.fused_dim + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, 
												 kernel_size=deconv_args['kernel_size'], stride=deconv_args['stride'], padding=deconv_args['padding']),
										  conv(self.fused_dim1 + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  nn.Conv2d(self.fused_dim1 + self.motion_out_dim, self.fused_dim1 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  ),
							nn.Sequential(nn.PReLU(self.fused_dim1 + self.motion_out_dim),
										  deconv(self.fused_dim1 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, 
												 kernel_size=deconv_args['kernel_size'], stride=deconv_args['stride'], padding=deconv_args['padding']),
										  conv(self.fused_dim2 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  nn.Conv2d(self.fused_dim2 + self.motion_out_dim, self.fused_dim2 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  ),
							nn.Sequential(nn.PReLU(self.fused_dim2 + self.motion_out_dim),
										  deconv(self.fused_dim2 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, 
												 kernel_size=deconv_args['kernel_size'], stride=deconv_args['stride'], padding=deconv_args['padding']),
										  conv(self.fused_dim3 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  nn.Conv2d(self.fused_dim3 + self.motion_out_dim, self.fused_dim3 + self.motion_out_dim, kernel_size=3, stride=1, padding=1),
										  ),
							])
		
		# residual refinement network
		in_chan = self.fused_dim3 + self.motion_out_dim + 15
		hidden_dim = 64
		# encoder
		self.proj = conv(in_chan, hidden_dim, kernel_size=3, stride=1, padding=1)
		self.down1 = nn.Sequential(
							conv(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1),
						)
		self.down2 = nn.Sequential(
							# concat with backbone decoder's (128-size) output first
							conv(self.fused_dim2 + hidden_dim, 2 * hidden_dim, kernel_size=3, stride=2, padding=1),
							conv(2 * hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		self.down3 = nn.Sequential(
							# concat with backbone decoder's (64-size) output first
							conv(self.fused_dim1 + 2 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=2, padding=1),
							conv(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
							conv(4 * hidden_dim, 4 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		# decoder
		self.up1 = nn.Sequential(
							deconv(4 * hidden_dim, 2 * hidden_dim, kernel_size=2, stride=2, padding=0),
							conv(2 * hidden_dim, 2 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		self.up2 = nn.Sequential(
							# concat with down2's output first
							deconv(4 * hidden_dim, 2 * hidden_dim, kernel_size=2, stride=2, padding=0),
							conv(2 * hidden_dim, 1 * hidden_dim, kernel_size=3, stride=1, padding=1),
						)
		self.up3 = nn.Sequential(
							# concat with down1's output first
							deconv(2 * hidden_dim, 1 * hidden_dim, kernel_size=2, stride=2, padding=0),
						)
		self.refine_head = nn.Sequential(
							# concat with proj's output first
							conv(2 * hidden_dim, 1 * hidden_dim, kernel_size=3, stride=1, padding=1),
							conv(1 * hidden_dim, 3, kernel_size=3, stride=1, padding=1),
						)

	def __set_local_window_size__(self, window_size):
		self.local_motion_args["window_size"] = window_size
		self.local_motion_atmformer[0]._set_window_size_(window_size, 0)
		self.local_motion_atmformer[1]._set_window_size_(window_size, window_size//2)

	def __set_global_window_size__(self, window_size):
		self.global_motion_args["window_size"] = window_size
		self.global_motion_atmformer[0]._set_window_size_(window_size, 0)
		self.global_motion_atmformer[1]._set_window_size_(window_size, window_size//2)

	def __freeze_global_motion__(self):
		self.last_feat_extract.requires_grad_(False)
		self.global_feature_fusion.requires_grad_(False)
		self.global_motion_atmformer.requires_grad_(False)
		self.global_motion_mlp.requires_grad_(False)

	def __finetune_global_motion__(self):
		self.last_feat_extract.requires_grad_(True)
		self.global_feature_fusion.requires_grad_(True)
		self.global_motion_atmformer.requires_grad_(True)
		self.global_motion_mlp.requires_grad_(True)

	def __freeze_local_motion__(self):
		self.feat_extracts.requires_grad_(False)
		self.cross_scale_feature_fusion.requires_grad_(False)
		self.local_motion_atmformer.requires_grad_(False)
		self.local_motion_mlp.requires_grad_(False)
		self.feat_enhance_transformer.requires_grad_(False)
		self.upsample_pyramid.requires_grad_(False)
		self.proj.requires_grad_(False)
		self.down1.requires_grad_(False)
		self.down2.requires_grad_(False)
		self.down3.requires_grad_(False)
		self.up1.requires_grad_(False)
		self.up2.requires_grad_(False)
		self.up3.requires_grad_(False)
		self.refine_head.requires_grad_(False)

	def __finetune_local_motion__(self):
		self.feat_extracts.requires_grad_(True)
		self.cross_scale_feature_fusion.requires_grad_(True)
		self.local_motion_atmformer.requires_grad_(True)
		self.local_motion_mlp.requires_grad_(True)
		self.feat_enhance_transformer.requires_grad_(True)
		self.upsample_pyramid.requires_grad_(True)
		self.proj.requires_grad_(True)
		self.down1.requires_grad_(True)
		self.down2.requires_grad_(True)
		self.down3.requires_grad_(True)
		self.up1.requires_grad_(True)
		self.up2.requires_grad_(True)
		self.up3.requires_grad_(True)
		self.refine_head.requires_grad_(True)

	def __finetune_refinenet_only__(self):
		self.last_feat_extract.requires_grad_(False)
		self.global_feature_fusion.requires_grad_(False)
		self.global_motion_atmformer.requires_grad_(False)
		self.global_motion_mlp.requires_grad_(False)
		self.feat_extracts.requires_grad_(False)
		self.cross_scale_feature_fusion.requires_grad_(False)
		self.local_motion_atmformer.requires_grad_(False)
		self.local_motion_mlp.requires_grad_(False)
		self.feat_enhance_transformer.requires_grad_(False)
		self.upsample_pyramid.requires_grad_(False)
		self.proj.requires_grad_(True)
		self.down1.requires_grad_(True)
		self.down2.requires_grad_(True)
		self.down3.requires_grad_(True)
		self.up1.requires_grad_(True)
		self.up2.requires_grad_(True)
		self.up3.requires_grad_(True)
		self.refine_head.requires_grad_(True)

	def forward(self, im0, im1):
		if not self.ensemble_global_motion:
			return self.forward_normal(im0, im1)
		else:
			return self.forward_global_ensemble(im0, im1)

	def shared_feat_extraction(self, x):
		'''
		x: torch.cat([im0, im1], dim=0), batch_size=2B for parallelism
		'''
		feat_scale_level = []
		for scale in range(self.pyramid_level):
			x = self.feat_extracts[scale](x)
			if scale != 0:
				feat_scale_level.append(x)

		return x, feat_scale_level
	
	def shared_feat_enhancement(self, x, h, w):
		'''
		feat: cross scale fusioned feature, shape [2B, HW, C]

		return: shape [2B, HW, C]
		'''
		x = einops.rearrange(x, 'B (H W) C -> B H W C', H=h, W=w)
		for k, blk in enumerate(self.feat_enhance_transformer):
			x = blk(x)
			if k % 2 == 0:
				x = einops.rearrange(x, 'B (H W) C -> B H W C', H=h)
		return x

	def estimate_local_motion(self, feat):
		'''
		feat: enhanced cross scale fusion feature, shape [2B, H, W, C]
		'''
		motion = []
		for k, blk in enumerate(self.local_motion_atmformer):
			B, h, w, _ = feat.size()
			feat, x_motion = blk(feat, h, w, B//2)
			if k == 0:
				feat = einops.rearrange(feat, 'B (H W) C -> B H W C', H=h)
			x_motion = einops.rearrange(x_motion, '(N B) L K -> B L (N K)', N=2)
			motion.append(x_motion)
		# [2*B, H*W, C] -> [B, 2*C, H, W]
		feat_concat = einops.rearrange(feat, '(N B) (H W) C-> B (N C) H W', N=2, H=h)
		motion = torch.cat(motion, dim=2)
		motion = einops.rearrange(motion, 'B (H W) C -> B C H W', H=h)

		out = self.local_motion_mlp(torch.cat([motion, feat_concat], dim=1))
		opt_flow_0 = out[:, :2]
		opt_flow_1 = out[:, 2:4]
		occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))

		return opt_flow_0, opt_flow_1, occ_mask1, feat, out
	
	def estimate_global_motion(self, x, feat_scale_level):
		feat_ = self.last_feat_extract(x)
		feat_scale_level.append(feat_)
		feat_scale_level.pop(0)
		feat_, h_, w_ = self.global_feature_fusion(feat_scale_level) # [B (HW) C]
		feat_ = einops.rearrange(feat_, 'B (H W) C -> B H W C', H=h_)

		motion = []
		for k, blk in enumerate(self.global_motion_atmformer):
			B, h_, w_, _ = feat_.size()
			feat_, x_motion = blk(feat_, h_, w_, B//2)
			if k == 0:
				feat_ = einops.rearrange(feat_, 'B (H W) C -> B H W C', H=h_)
			x_motion = einops.rearrange(x_motion, '(N B) L K -> B L (N K)', N=2)
			motion.append(x_motion)
		# [2*B, H*W, C] -> [B, 2*C, H, W]
		feat_ = einops.rearrange(feat_, '(N B) (H W) C-> B (N C) H W', N=2, H=h_)
		motion = torch.cat(motion, dim=2)
		motion = einops.rearrange(motion, 'B (H W) C -> B C H W', H=h_)
		out = self.global_motion_mlp(torch.cat([motion, feat_], dim=1))
		opt_flow_0 = out[:, :2]
		opt_flow_1 = out[:, 2:4]
		occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))

		return opt_flow_0, opt_flow_1, occ_mask1
	
	def residual_refinement(self, feat, im0, I_t_0, im1, I_t_1, I_t, backbone_decoder_feats):
		feat0 = torch.cat([feat, im0, I_t_0, im1, I_t_1, I_t], dim=1) 
		feat0 = self.proj(feat0)
		feat1 = self.down1(feat0)  
		feat2 = self.down2(torch.cat([feat1, backbone_decoder_feats.pop()], dim=1)) 
		feat3 = self.down3(torch.cat([feat2, backbone_decoder_feats.pop()], dim=1)) 
		# decoder
		feat2_ = self.up1(feat3) # 128
		feat1_ = self.up2(torch.cat([feat2_, feat2], dim=1))
		feat0_ = self.up3(torch.cat([feat1_, feat1], dim=1))
		# output
		I_t_residual = self.refine_head(torch.cat([feat0_, feat0], dim=1)) 
		I_t_residual = 2*torch.sigmoid(I_t_residual) - 1  # mapped to [-1, 1]

		return I_t_residual
	
	def forward_normal(self, im0, im1):
		'''
		im0, im1: tensor [B,3,H,W], float32, normalized to [0, 1]
		'''
		B,_,H,W = im0.size()
		im0_list = [im0]
		im1_list = [im1]
		im_t_list = []
		im0_warped_list = []
		im1_warped_list = []
		# downscale input frames
		for scale in range(self.pyramid_level-1):
			im0_down = F.interpolate(im0_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
			im1_down = F.interpolate(im1_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
			im0_list.append(im0_down)
			im1_list.append(im1_down)

		# CNN feature extraction
		feat_ = torch.cat([im0, im1], dim=0) # speed up using parallelization
		feat_, feat_scale_level = self.shared_feat_extraction(feat_)

		# cross scale feature fusion
		feat, h, w = self.cross_scale_feature_fusion(feat_scale_level) # [2B (HW) C]

		if self.global_motion:
			opt_flow_0, opt_flow_1, occ_mask1 = self.estimate_global_motion(feat_, feat_scale_level)
			occ_mask2 = 1 - occ_mask1

			im0_down = F.interpolate(im0_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
			im1_down = F.interpolate(im1_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)

			I_t_0 = flow_warp(im0_down, opt_flow_0)
			I_t_1 = flow_warp(im1_down, opt_flow_1)
			I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1
			im0_warped_list.insert(0, I_t_0)
			im1_warped_list.insert(0, I_t_1)
			im_t_list.insert(0, I_t)

			opt_flow_0 = upsample_flow(opt_flow_0, upsample_factor=2, mode='bilinear')
			opt_flow_1 = upsample_flow(opt_flow_1, upsample_factor=2, mode='bilinear')
			
			feat = einops.rearrange(feat, 'B (H W) C -> B C H W', H=h)
			feat0 = flow_warp(feat[:B], flow=opt_flow_0)
			feat1 = flow_warp(feat[B:], flow=opt_flow_1)
			feat = torch.cat([feat0, feat1], dim=0)
			feat = einops.rearrange(feat, 'B C H W -> B H W C', H=h)

			for i in reversed(range(self.pyramid_level)):
				im0_list[i] = flow_warp(im0_list[i], flow=opt_flow_0)
				im1_list[i] = flow_warp(im1_list[i], flow=opt_flow_1)
				if i != 0:
					opt_flow_0 = upsample_flow(opt_flow_0, upsample_factor=2, mode='bilinear')
					opt_flow_1 = upsample_flow(opt_flow_1, upsample_factor=2, mode='bilinear')

		else:
			feat = einops.rearrange(feat, 'B (H W) C -> B H W C', H=h)

		opt_flow_0, opt_flow_1, occ_mask1, feat, out = self.estimate_local_motion(feat)
		occ_mask2 = 1 - occ_mask1

		feat = self.shared_feat_enhancement(feat, h, w)
		feat = einops.rearrange(feat, '(N B) (H W) C -> B (N C) H W', N=2, H=h)

		I_t_0 = flow_warp(im0_list[-1], opt_flow_0)
		I_t_1 = flow_warp(im1_list[-1], opt_flow_1)
		I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1
		im0_warped_list.insert(0, I_t_0)
		im1_warped_list.insert(0, I_t_1)
		im_t_list.insert(0, I_t)
		
		# only warped feature once, 只warp一次、剩下的自己看著辦
		feat1 = flow_warp(feat[:, :self.fused_dims[0]], opt_flow_0)
		feat2 = flow_warp(feat[:, self.fused_dims[0]:self.fused_dims[-1]], opt_flow_1)
		feat = torch.cat([feat1, feat2, out], dim=1)

		backbone_decoder_feats = []

		# upscale motion along with feature
		for i, scale in enumerate(reversed(range(self.pyramid_level-1))):
			# forward features to get finer resolution
			feat = self.upsample_pyramid[i](feat) 
			out = feat[:, -self.motion_out_dim:] 
			opt_flow_0 = out[:, :2]
			opt_flow_1 = out[:, 2:4]
			occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))
			occ_mask2 = 1 - occ_mask1

			if scale != 0:
				backbone_decoder_feats.append(feat[:, :-self.motion_out_dim] )

			I_t_0 = flow_warp(im0_list[scale], opt_flow_0)
			I_t_1 = flow_warp(im1_list[scale], opt_flow_1)
			I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1
			im0_warped_list.insert(0, I_t_0)
			im1_warped_list.insert(0, I_t_1)
			im_t_list.insert(0, I_t)

		# residual refinement 
		I_t_residual = self.residual_refinement(feat, im0, I_t_0, im1, I_t_1, I_t, backbone_decoder_feats)
		I_t += I_t_residual
		I_t = torch.clamp(I_t, 0, 1)

		output_dict = {'I_t': I_t, 
				 	   'im_t_list': im_t_list, # scale: fine to coarse
					   'im0_warped_list': im0_warped_list,
					   'im1_warped_list': im1_warped_list,
					   'opt_flow_0': opt_flow_0,
					   'opt_flow_1': opt_flow_1,
					   'I_t_0': I_t_0,
					   'I_t_1': I_t_1,
					   'occ_mask1': occ_mask1,
					   'occ_mask2': occ_mask2,
					   }
		return output_dict

	def global_alignmentness(self, x, im0, im1):
		opt_flow0 = x[0]
		opt_flow1 = x[1]
		_,_,H1,W1 = opt_flow0.size()
		_,_,H0,W0 = im0.size()

		upsample_factor = H0 // H1
		opt_flow0 = upsample_flow(opt_flow0, upsample_factor=upsample_factor, mode='bilinear')
		opt_flow1 = upsample_flow(opt_flow1, upsample_factor=upsample_factor, mode='bilinear')
		im0 = flow_warp(im0, opt_flow0)
		im1 = flow_warp(im1, opt_flow1)

		l1 = nn.L1Loss(reduce=False, reduction='none')
		loss = torch.mean(l1(im0, im1), dim=[1,2,3])
		return loss
	
	def multiscale_global_motion_ensemble(self, im0, im1):
		'''
		im0, im1: tensor [B,3,H,W], float32, normalized to [0, 1]
		'''
		B,_,_,_ = im0.size()
		im = torch.cat([im0, im1], dim=0) # speed up using parallelization

		# original scale (H,W)
		feat_, feat_scale_level = self.shared_feat_extraction(im)
		level0 = self.estimate_global_motion(feat_, feat_scale_level)
		opt_flow0_l0, opt_flow1_l0 = level0[0], level0[1]
		loss0 = self.global_alignmentness(level0, im0, im1)

		# downscale level 1 (H/2, W/2)
		im = F.interpolate(im, scale_factor=0.5, mode='bilinear', align_corners=True)
		feat_, feat_scale_level = self.shared_feat_extraction(im)
		level1 = self.estimate_global_motion(feat_, feat_scale_level)
		opt_flow0_l1, opt_flow1_l1 = level1[0], level1[1]
		loss1 = self.global_alignmentness(level1, im0, im1)

		# downscale level 2 (W/4, W/4)
		im = F.interpolate(im, scale_factor=0.5, mode='bilinear', align_corners=True)
		feat_, feat_scale_level = self.shared_feat_extraction(im)
		level2 = self.estimate_global_motion(feat_, feat_scale_level)
		opt_flow0_l2, opt_flow1_l2 = level2[0], level2[1]
		loss2 = self.global_alignmentness(level2, im0, im1)

		opt_flow0 = torch.zeros_like(opt_flow0_l0)
		opt_flow1 = torch.zeros_like(opt_flow1_l0)
		for i in range(B):
			min_loss = min(loss0[i], loss1[i], loss2[i])
			if loss0[i] == min_loss:
				opt_flow0[i] = opt_flow0_l0[i]
				opt_flow1[i] = opt_flow1_l0[i]
			elif loss1[i] == min_loss:
				opt_flow0[i] = upsample_flow(opt_flow0_l1[i,None], upsample_factor=2, mode='bilinear')
				opt_flow1[i] = upsample_flow(opt_flow1_l1[i,None], upsample_factor=2, mode='bilinear')
			else:
				opt_flow0[i] = upsample_flow(opt_flow0_l2[i,None], upsample_factor=4, mode='bilinear')
				opt_flow1[i] = upsample_flow(opt_flow1_l2[i,None], upsample_factor=4, mode='bilinear')
		
		return opt_flow0, opt_flow1

	def forward_global_ensemble(self, im0, im1):
		'''
		im0, im1: tensor [B,3,H,W], float32, normalized to [0, 1]
		'''
		B,_,H,W = im0.size()
		im0_list = [im0]
		im1_list = [im1]
		im_t_list = []
		im0_warped_list = []
		im1_warped_list = []
		# downscale input frames
		for scale in range(self.pyramid_level-1):
			im0_down = F.interpolate(im0_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
			im1_down = F.interpolate(im1_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
			im0_list.append(im0_down)
			im1_list.append(im1_down)

		# CNN feature extraction
		feat_ = torch.cat([im0, im1], dim=0) # speed up using parallelization
		feat_, feat_scale_level = self.shared_feat_extraction(feat_)

		# cross scale feature fusion
		feat, h, w = self.cross_scale_feature_fusion(feat_scale_level) # [B (HW) C]

		if self.global_motion:
			opt_flow_0, opt_flow_1 = self.multiscale_global_motion_ensemble(im0, im1)

			im0_down = F.interpolate(im0_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)
			im1_down = F.interpolate(im1_list[-1], scale_factor=0.5, mode='bilinear', align_corners=True)

			opt_flow_0 = upsample_flow(opt_flow_0, upsample_factor=2, mode='bilinear')
			opt_flow_1 = upsample_flow(opt_flow_1, upsample_factor=2, mode='bilinear')
			
			feat = einops.rearrange(feat, 'B (H W) C -> B C H W', H=h)
			feat0 = flow_warp(feat[:B], flow=opt_flow_0)
			feat1 = flow_warp(feat[B:], flow=opt_flow_1)
			feat = torch.cat([feat0, feat1], dim=0)
			feat = einops.rearrange(feat, 'B C H W -> B H W C', H=h)

			for i in reversed(range(self.pyramid_level)):
				im0_list[i] = flow_warp(im0_list[i], flow=opt_flow_0)
				im1_list[i] = flow_warp(im1_list[i], flow=opt_flow_1)
				if i != 0:
					opt_flow_0 = upsample_flow(opt_flow_0, upsample_factor=2, mode='bilinear')
					opt_flow_1 = upsample_flow(opt_flow_1, upsample_factor=2, mode='bilinear')

		else:
			feat = einops.rearrange(feat, 'B (H W) C -> B H W C', H=h)

		opt_flow_0, opt_flow_1, occ_mask1, feat, out = self.estimate_local_motion(feat)
		occ_mask2 = 1 - occ_mask1

		feat = self.shared_feat_enhancement(feat, h, w)
		feat = einops.rearrange(feat, '(N B) (H W) C -> B (N C) H W', N=2, H=h)

		I_t_0 = flow_warp(im0_list[-1], opt_flow_0)
		I_t_1 = flow_warp(im1_list[-1], opt_flow_1)
		I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1
		im0_warped_list.insert(0, I_t_0)
		im1_warped_list.insert(0, I_t_1)
		im_t_list.insert(0, I_t)
		
		# only warped feature once, 只warp一次、剩下的自己看著辦
		feat1 = flow_warp(feat[:, :self.fused_dims[0]], opt_flow_0)
		feat2 = flow_warp(feat[:, self.fused_dims[0]:self.fused_dims[-1]], opt_flow_1)
		feat = torch.cat([feat1, feat2, out], dim=1)

		backbone_decoder_feats = []

		# upscale motion along with feature
		for i, scale in enumerate(reversed(range(self.pyramid_level-1))):
			# forward features to get finer resolution
			feat = self.upsample_pyramid[i](feat) 
			out = feat[:, -self.motion_out_dim:] 
			opt_flow_0 = out[:, :2]
			opt_flow_1 = out[:, 2:4]
			occ_mask1 = torch.sigmoid(out[:, 4].unsqueeze(1))
			occ_mask2 = 1 - occ_mask1

			if scale != 0:
				backbone_decoder_feats.append(feat[:, :-self.motion_out_dim] )

			I_t_0 = flow_warp(im0_list[scale], opt_flow_0)
			I_t_1 = flow_warp(im1_list[scale], opt_flow_1)
			I_t = occ_mask1 * I_t_0 + occ_mask2 * I_t_1
			im0_warped_list.insert(0, I_t_0)
			im1_warped_list.insert(0, I_t_1)
			im_t_list.insert(0, I_t)

		# residual refinement 
		I_t_residual = self.residual_refinement(feat, im0, I_t_0, im1, I_t_1, I_t, backbone_decoder_feats)
		I_t += I_t_residual
		I_t = torch.clamp(I_t, 0, 1)

		output_dict = {'I_t': I_t, 
				 	   'im_t_list': im_t_list, # scale: fine to coarse
					   'im0_warped_list': im0_warped_list,
					   'im1_warped_list': im1_warped_list,
					   'opt_flow_0': opt_flow_0,
					   'opt_flow_1': opt_flow_1,
					   'I_t_0': I_t_0,
					   'I_t_1': I_t_1,
					   'occ_mask1': occ_mask1,
					   'occ_mask2': occ_mask2,
					   }
		return output_dict

	
if __name__ == "__main__":
	# device = torch.device('cpu')
	device = torch.device('cuda')

	model = Network().to(device)
	pytorch_total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
	print(f"total trainable parameters: {round(pytorch_total_params/1e6, 2)} M") 
	
	model.global_motion = True
	# model.global_motion = False

	criterion = nn.MSELoss()
	optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
	batchsize = 4

	a = torch.rand(batchsize,3,256,256).to(device)
	b = torch.rand(batchsize,3,256,256).to(device)
	c = torch.rand(batchsize,3,256,256).to(device)

	for i in range(10):
		print(i)
		c_logits = model(a,b)
		loss = criterion(c_logits, c)
		loss.backward()
		optimizer.step()
