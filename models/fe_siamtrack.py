
import torch.nn.functional as F
import torch.nn.init
import numpy as np
import torch
import cv2
from pathlib import Path
from models.common import *
from models.template import Template
from utils.losses import *
import matplotlib
matplotlib.use("Agg")  # safe in headless
import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import torchvision.transforms.functional as TVF
import scipy.io as sio
import os

class FPNEncoder(nn.Module):
    def __init__(self, in_channels=1, out_channels=512, recurrent=False):
        super(FPNEncoder, self).__init__()
        self.conv_bottom_1 = ConvBlock(
            in_channels=32,
            out_channels=64,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_2 = ConvBlock(
            in_channels=64,
            out_channels=128,
            n_convs=2,
            kernel_size=5,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_3 = ConvBlock(
            in_channels=128,
            out_channels=256,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=True,
        )
        self.conv_bottom_4 = ConvBlock(
            in_channels=256,
            out_channels=out_channels,
            n_convs=2,
            kernel_size=3,
            padding=0,
            downsample=False,
        )
        self.recurrent = recurrent
        if self.recurrent:
            self.conv_rnn = ConvLSTMCell(out_channels, out_channels, 1)
        self.conv_lateral_3 = nn.Conv2d(
            in_channels=256, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_2 = nn.Conv2d(
            in_channels=128, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_1 = nn.Conv2d(
            in_channels=64, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_lateral_0 = nn.Conv2d(
            in_channels=32, out_channels=out_channels, kernel_size=1, bias=True
        )
        self.conv_dealias_3 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_1 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_dealias_0 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=3,
            padding=1,
            bias=True,
        )
        self.conv_out = nn.Sequential(
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_convs=1,
                kernel_size=3,
                padding=1,
                downsample=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )
        self.conv_bottleneck_out = nn.Sequential(
            ConvBlock(
                in_channels=out_channels,
                out_channels=out_channels,
                n_convs=1,
                kernel_size=3,
                padding=1,
                downsample=False,
            ),
            nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1,
                bias=True,
            ),
        )
    def reset(self):
        if self.recurrent:
            self.conv_rnn.reset()
    def forward(self, c0):
        """
        :param x:
        :return: (highest res feature map, lowest res feature map)
        """
        # Bottom-up pathway
        #c0 = self.conv_bottom_0(x)  # 31x31
        c1 = self.conv_bottom_1(c0)  # 23x23
        c2 = self.conv_bottom_2(c1)  # 15x15
        c3 = self.conv_bottom_3(c2)  # 5x5
        c4 = self.conv_bottom_4(c3)  # 1x1
        # Top-down pathway (with lateral cnx and de-aliasing)
        p4 = c4
        p3 = self.conv_dealias_3(
            self.conv_lateral_3(c3)
            + F.interpolate(p4, (c3.shape[2], c3.shape[3]), mode="bilinear")
        )
        p2 = self.conv_dealias_2(
            self.conv_lateral_2(c2)
            + F.interpolate(p3, (c2.shape[2], c2.shape[3]), mode="bilinear")
        )
        p1 = self.conv_dealias_1(
            self.conv_lateral_1(c1)
            + F.interpolate(p2, (c1.shape[2], c1.shape[3]), mode="bilinear")
        )
        p0 = self.conv_dealias_0(
            self.conv_lateral_0(c0)
            + F.interpolate(p1, (c0.shape[2], c0.shape[3]), mode="bilinear")
        )
        if self.recurrent:
            p0 = self.conv_rnn(p0)
        return self.conv_out(p0), self.conv_bottleneck_out(c4)

class SiameseFPN(nn.Module):
    """
    One FPN, two inputs (events, frames). Weights are shared.
    If frames have different #channels than events, we adapt them with 1x1 conv.
    """
    def __init__(self,
                 in_ch_events: int,
                 in_ch_frames: int,
                 out_channels: int = 512,
                 recurrent: bool = False):
        super().__init__()
        self.conv_bottom_0_events = ConvBlock(
            in_channels=in_ch_events,
            out_channels=32,
            n_convs=2,
            kernel_size=1,
            padding=0,
            downsample=False,
        )
        self.conv_bottom_0_frames = ConvBlock(
            in_channels=in_ch_frames,
            out_channels=32,
            n_convs=2,
            kernel_size=1,
            padding=0,
            downsample=False,
        )
        # this is the ONLY FPN → shared weights
        self.shared_fpn = FPNEncoder(in_channels=in_ch_events,
                                     out_channels=out_channels,
                                     recurrent=recurrent)
        # if in_ch_frames != in_ch_events:
        #     self.frames_adapter = nn.Conv2d(in_ch_frames, in_ch_events, kernel_size=1, bias=True)
        # else:
        #     self.frames_adapter = None
    def reset(self):
        if self.shared_fpn.recurrent:
            self.shared_fpn.conv_rnn.reset()
    def encode_events(self, x_e):
        c0 = self.conv_bottom_0_events(x_e)          # (B,32,31,31)
        return self.shared_fpn(c0)
    def encode_frames(self, x_f):
        c0 = self.conv_bottom_0_frames(x_f)          # (B,32,31,31)
        return self.shared_fpn(c0)
    def forward(self, x_events, x_frames):
        f_e, d_e = self.encode_events(x_events)
        f_f, d_f = self.encode_frames(x_frames)
        return f_e, d_e, f_f, d_f
class JointEncoder(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(JointEncoder, self).__init__()
        self.conv1 = ConvBlock(
            in_channels=129, out_channels=128, n_convs=2, downsample=True
        )
        self.conv2 = ConvBlock(
            in_channels=128, out_channels=128, n_convs=2, downsample=True
        )
        self.convlstm0 = ConvLSTMCell(128, 128, 3)
        self.conv3 = ConvBlock(
            in_channels=128, out_channels=256, n_convs=2, downsample=True
        )
        self.conv4 = ConvBlock(
            in_channels=256,
            out_channels=256,
            kernel_size=3,
            padding=0,
            n_convs=1,
            downsample=False,
        )
        # ---- Token attention over tracks ----
        self.flatten = nn.Flatten()
        embed_dim = 256
        num_heads = 8
        self.multihead_attention0 = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )
        # NEW: LayerNorm + Dropout around attention
        self.pre_attn_ln = nn.LayerNorm(embed_dim)
        self.attn_drop = nn.Dropout(p=0.1)
        self.prev_x_res = None
        self.gates = nn.Linear(2 * embed_dim, embed_dim)
        self.ls_layer = LayerScale(embed_dim)
        # Feature+state fusion (keep)
        self.fusion_layer0 = nn.Sequential(
            nn.Linear(embed_dim * 2, embed_dim),
            nn.LeakyReLU(0.1),
            nn.Linear(embed_dim, embed_dim),
            nn.LeakyReLU(0.1),
        )
        self.output_layers = nn.Sequential(
            nn.Linear(embed_dim, 512),
            nn.LeakyReLU(0.1)
        )
    def reset(self):
        self.convlstm0.reset()
        self.prev_x_res = None
    def forward(self, x, attn_mask=None):
        # conv feature encoder
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.convlstm0(x)
        x = self.conv3(x)
        x = self.conv4(x)
        # (K, D) where K = #tracks in batch, D = 256
        x = self.flatten(x)
        if self.prev_x_res is None:
            self.prev_x_res = Variable(torch.zeros_like(x))
        # fuse current token with previous state (keep)
        x = self.fusion_layer0(torch.cat((x, self.prev_x_res), dim=1))  # (K, D)
        # ---- Attention over tracks (NO detach) ----
        # MHA expects (N, S, E) with batch_first=True
        # Your original code used N=1, S=K, E=D
        tokens = self.pre_attn_ln(x).unsqueeze(0)  # (1, K, D)
        if self.training:
            # keep EXACT same mask behavior as your current code:
            # only apply mask during training, and cast to bool
            if attn_mask is not None:
                attn_out = self.multihead_attention0(
                    query=tokens,
                    key=tokens,
                    value=tokens,
                    attn_mask=attn_mask.bool(),
                )[0]
            else:
                attn_out = self.multihead_attention0(
                    query=tokens, key=tokens, value=tokens
                )[0]
        else:
            # keep same inference behavior as your current code: no mask
            attn_out = self.multihead_attention0(
                query=tokens, key=tokens, value=tokens
            )[0]
        attn_out = attn_out.squeeze(0)  # (K, D)
        # residual with LayerScale + dropout (NEW)
        x = x + self.ls_layer(self.attn_drop(attn_out))
        # gated state update (keep)
        gate_weight = torch.sigmoid(self.gates(torch.cat((self.prev_x_res, x), dim=1)))
        x = self.prev_x_res * gate_weight + x * (1.0 - gate_weight)
        self.prev_x_res = x
        # project to 512 (keep)
        x = self.output_layers(x)
        return x
    
class LayerScale(nn.Module):
    def __init__(self, dim, init_values=1e-5, inplace=False):
        super().__init__()
        self.inplace = inplace
        self.gamma = nn.Parameter(init_values * torch.ones(dim))
    
    def forward(self, x):
        gamma = self.gamma
        return x.mul_(gamma) if self.inplace else x * gamma
    
    def gate(self):
        # g in [0,1], same shape as gamma: (dim,)
        return torch.sigmoid(self.gamma)
    
class TrackerNetCSiameseSeperateC0(Template):
    def __init__(
        self,
        representation="time_surfaces_1",
        max_unrolls=16,
        n_vis=8,
        feature_dim=1024,
        patch_size=31,
        init_unrolls=1,
        input_channels=None,
        **kwargs,
        
    ):
        super(TrackerNetCSiameseSeperateC0, self).__init__(
            representation=representation,
            max_unrolls=max_unrolls,
            init_unrolls=init_unrolls,
            n_vis=n_vis,
            patch_size=patch_size,
            **kwargs,
        )
        # Configuration
        self.grayscale_ref = True
        if not isinstance(input_channels, type(None)):
            self.channels_in_per_patch = input_channels
        
        self.tb_hist_every = 50
        # Architecture
        self.feature_dim = feature_dim
        self.redir_dim = 128
        self.gradlog_weight = 1.0  # or 0.1 to start safer
        self.grad_logI_center = None
        self.save_rec_grad = True
        self.rec_grad_dir = Path(__file__).resolve().parents[1] / "rec_grad"
        self.rec_grad_max_save = 4
        self.rec_grad_counter = 0
        
        # BEFORE: 2 separate FPNs
        # self.reference_encoder = FPNEncoder(1, self.feature_dim)
        # self.target_encoder    = FPNEncoder(self.channels_in_per_patch, self.feature_dim)
        # AFTER: ONE shared FPN
        #   events:  Ce = self.channels_in_per_patch
        #   frames:  Cf = 1  (mag + phase)
        self.siamese_fpn = SiameseFPN(
            in_ch_events=self.channels_in_per_patch,
            in_ch_frames=1,                 # <- frame side is 2ch (mag,phase)
            out_channels=self.feature_dim,
            recurrent=False,
        )
        # Correlation3 had k=1, p=0
        self.reference_redir = nn.Conv2d(
            self.feature_dim, self.redir_dim, kernel_size=3, padding=1
        )
        self.target_redir = nn.Conv2d(
            self.feature_dim, self.redir_dim, kernel_size=3, padding=1
        )
        self.softmax = nn.Softmax(dim=2)
        self.joint_encoder = JointEncoder(
            in_channels=1 + 2 * self.redir_dim, out_channels=512
        )
        self.predictor = nn.Linear(in_features=512, out_features=2, bias=False)
        self.flatten = nn.Flatten()
        # Operational
        #self.loss = L1Truncated(patch_size=patch_size)
        #self.loss = L1TruncatedWithLogGrad(patch_size=patch_size, grad_weight=self.gradlog_weight)
        self.name = f"corr_{self.representation}"
        # Persistent Tensors
        self.f_ref, self.d_ref = None, None
        self.correlation_maps = []
        self.inputs = []
        self.refs = []
        self.saved_x_frames_original = False  # <--- add this
        self.sobel_eps = 1e-12
        self.saved_frame_repr_mat = False
        self.frame_collapse = nn.Conv2d(
            in_channels=2,
            out_channels=1,
            kernel_size=1,
            bias=True
        )
        # Optional init: equal weights → like averaging gx, gy
        with torch.no_grad():
            self.frame_collapse.weight[:] = 1.0 / np.sqrt(2.0)
            if self.frame_collapse.bias is not None:
                self.frame_collapse.bias.zero_()
    
    def init_weights(self):
        torch.nn.init.xavier_uniform(self.fc_out.weight)
    
    def reset(self, _):
        self.d_ref, self.f_ref = None, None
        self.joint_encoder.reset()
        self.grad_logI_center = None   # <- NEW
      
    def forward(self, x, attn_mask=None):
        """
        x: (B, Ce+Cf, H, W)
        first Ce channels: events
        remaining Cf: raw frame patch
        """
        x_events = x[:, : self.channels_in_per_patch, :, :]   # (B, Ce, H, W)
        x_frames = x[:, self.channels_in_per_patch :, :, :]   # (B, Cf, H, W)
        # --- encode events with SiameseFPN ---
        f_ev, d_ev = self.siamese_fpn.encode_events(x_events)
        # --- encode reference frames once, cache f_ref & d_ref ---
        if self.f_ref is None:
            f_fr, d_fr = self.siamese_fpn.encode_frames(x_frames)
            self.f_ref = self.reference_redir(f_fr)
            self.d_ref = d_fr
        # --- correlation ---
        f_corr = (f_ev * self.d_ref).sum(dim=1, keepdim=True)  # (B,1,H,W)
        f_corr = self.softmax(
            f_corr.view(-1, 1, self.patch_size * self.patch_size)
        ).view(-1, 1, self.patch_size, self.patch_size)
        # --- joint encoding & prediction ---
        f = torch.cat([f_corr, self.target_redir(f_ev)], dim=1)
        f = self.joint_encoder(f, attn_mask)
        f = self.predictor(f)
        return f
