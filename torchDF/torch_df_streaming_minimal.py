"""
ONNX exportable classes
"""

import math
import torch
import argparse
import torchaudio

from torch.nn import functional as F

from torch import nn
from torch import Tensor
from typing import Tuple, List

from df import init_df

from functools import partial
from typing import Callable, List, Optional, Tuple

import torch
from torch import Tensor, nn
import torch.nn.functional as F

from df.config import Csv, DfParams, config
from df.modules import Conv2dNormAct, ConvTranspose2dNormAct

from typing_extensions import Final
from torch.nn.parameter import Parameter
from torch.nn import init
import numpy as np


from torch.autograd import Function


class OnnxComplexMul(Function):
    """Auto-grad function to mimic irfft for ONNX exporting"""

    @staticmethod
    def forward(ctx, input_0: torch.Tensor, input_1: torch.Tensor) -> torch.Tensor:
        return torch.view_as_real(
            torch.view_as_complex(input_0) * torch.view_as_complex(input_1)
        )

    @staticmethod
    def symbolic(
        g: torch.Graph, input_0: torch.Value, input_1: torch.Value
    ) -> torch.Value:
        """Symbolic representation for onnx graph"""
        return g.op("ai.onnx.contrib::ComplexMul", input_0, input_1)


class SqueezedGRU_S(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: Optional[int] = None,
        num_layers: int = 1,
        linear_groups: int = 8,
        batch_first: bool = True,
        gru_skip_op: Optional[Callable[..., torch.nn.Module]] = None,
        linear_act_layer: Callable[..., torch.nn.Module] = nn.Identity,
    ):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.linear_in = nn.Sequential(
            GroupedLinearEinsum(
                input_size, hidden_size, linear_groups, linear_act_layer()
            ),
        )
        self.gru = nn.GRU(
            hidden_size, hidden_size, num_layers=num_layers, batch_first=batch_first
        )
        self.gru_skip = gru_skip_op() if gru_skip_op is not None else None
        if output_size is not None:
            self.linear_out = nn.Sequential(
                GroupedLinearEinsum(
                    hidden_size, output_size, linear_groups, linear_act_layer()
                ),
            )
        else:
            self.linear_out = nn.Identity()

    def forward(self, input: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
        x = self.linear_in(input)
        x, h = self.gru(x, h)
        x = self.linear_out(x)
        if self.gru_skip is not None:
            x = x + self.gru_skip(input)
        return x, h


class GroupedLinearEinsum(nn.Module):
    input_size: Final[int]
    hidden_size: Final[int]
    groups: Final[int]

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        groups: int = 1,
        activation=nn.Identity(),
    ):
        super().__init__()
        # self.weight: Tensor
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.groups = groups
        assert (
            input_size % groups == 0
        ), f"Input size {input_size} not divisible by {groups}"
        assert (
            hidden_size % groups == 0
        ), f"Hidden size {hidden_size} not divisible by {groups}"
        self.ws = input_size // groups
        self.register_parameter(
            "weight",
            Parameter(
                torch.zeros(groups, input_size // groups, hidden_size // groups),
                requires_grad=True,
            ),
        )
        self.reset_parameters()
        self.activation = activation

    def reset_parameters(self):
        init.kaiming_uniform_(self.weight, a=math.sqrt(5))  # type: ignore

    def forward(self, x: Tensor) -> Tensor:
        x = x.reshape(self.groups, 1, self.ws)
        x = torch.matmul(x, self.weight)
        x = self.activation(x)
        return x.view(1, 1, -1)

    def __repr__(self):
        cls = self.__class__.__name__
        return f"{cls}(input_size: {self.input_size}, hidden_size: {self.hidden_size}, groups: {self.groups})"


class ModelParams(DfParams):
    section = "deepfilternet"

    def __init__(self):
        super().__init__()
        self.conv_lookahead: int = config(
            "CONV_LOOKAHEAD", cast=int, default=0, section=self.section
        )
        self.conv_ch: int = config(
            "CONV_CH", cast=int, default=16, section=self.section
        )
        self.conv_depthwise: bool = config(
            "CONV_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.convt_depthwise: bool = config(
            "CONVT_DEPTHWISE", cast=bool, default=True, section=self.section
        )
        self.conv_kernel: List[int] = config(
            "CONV_KERNEL",
            cast=Csv(int),
            default=(1, 3),
            section=self.section,  # type: ignore
        )
        self.convt_kernel: List[int] = config(
            "CONVT_KERNEL",
            cast=Csv(int),
            default=(1, 3),
            section=self.section,  # type: ignore
        )
        self.conv_kernel_inp: List[int] = config(
            "CONV_KERNEL_INP",
            cast=Csv(int),
            default=(3, 3),
            section=self.section,  # type: ignore
        )
        self.emb_hidden_dim: int = config(
            "EMB_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.emb_num_layers: int = config(
            "EMB_NUM_LAYERS", cast=int, default=2, section=self.section
        )
        self.emb_gru_skip_enc: str = config(
            "EMB_GRU_SKIP_ENC", default="none", section=self.section
        )
        self.emb_gru_skip: str = config(
            "EMB_GRU_SKIP", default="none", section=self.section
        )
        self.df_hidden_dim: int = config(
            "DF_HIDDEN_DIM", cast=int, default=256, section=self.section
        )
        self.df_gru_skip: str = config(
            "DF_GRU_SKIP", default="none", section=self.section
        )
        self.df_pathway_kernel_size_t: int = config(
            "DF_PATHWAY_KERNEL_SIZE_T", cast=int, default=1, section=self.section
        )
        self.enc_concat: bool = config(
            "ENC_CONCAT", cast=bool, default=False, section=self.section
        )
        self.df_num_layers: int = config(
            "DF_NUM_LAYERS", cast=int, default=3, section=self.section
        )
        self.df_n_iter: int = config(
            "DF_N_ITER", cast=int, default=1, section=self.section
        )
        self.lin_groups: int = config(
            "LINEAR_GROUPS", cast=int, default=1, section=self.section
        )
        self.enc_lin_groups: int = config(
            "ENC_LINEAR_GROUPS", cast=int, default=16, section=self.section
        )
        self.mask_pf: bool = config(
            "MASK_PF", cast=bool, default=False, section=self.section
        )
        self.lsnr_dropout: bool = config(
            "LSNR_DROPOUT", cast=bool, default=False, section=self.section
        )


class Add(nn.Module):
    def forward(self, a, b):
        return a + b


class Concat(nn.Module):
    def forward(self, a, b):
        return torch.cat((a, b), dim=-1)


class Encoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 4 == 0, "erb_bins should be divisible by 4"

        self.erb_conv0 = Conv2dNormAct(
            1, p.conv_ch, kernel_size=p.conv_kernel_inp, bias=False, separable=True
        )
        self.conv_buffer_size = p.conv_kernel_inp[0] - 1
        self.conv_ch = p.conv_ch

        conv_layer = partial(
            Conv2dNormAct,
            in_ch=p.conv_ch,
            out_ch=p.conv_ch,
            kernel_size=p.conv_kernel,
            bias=False,
            separable=True,
        )
        self.erb_conv1 = conv_layer(fstride=2)
        self.erb_conv2 = conv_layer(fstride=2)
        self.erb_conv3 = conv_layer(fstride=1)
        self.df_conv0_ch = p.conv_ch
        self.df_conv0 = Conv2dNormAct(
            2,
            self.df_conv0_ch,
            kernel_size=p.conv_kernel_inp,
            bias=False,
            separable=True,
        )
        self.df_conv1 = conv_layer(fstride=2)
        self.erb_bins = p.nb_erb
        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4
        df_fc_emb = GroupedLinearEinsum(
            p.conv_ch * p.nb_df // 2,
            self.emb_in_dim,
            groups=p.enc_lin_groups,
            activation=nn.ReLU(inplace=True),
        )
        self.df_fc_emb = nn.Sequential(df_fc_emb)
        if p.enc_concat:
            self.emb_in_dim *= 2
            self.combine = Concat()
        else:
            self.combine = Add()
        self.emb_n_layers = p.emb_num_layers
        if p.emb_gru_skip_enc == "none":
            skip_op = None
        elif p.emb_gru_skip_enc == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif p.emb_gru_skip_enc == "groupedlinear":
            skip_op = partial(
                GroupedLinearEinsum,
                input_size=self.emb_out_dim,
                hidden_size=self.emb_out_dim,
                groups=p.lin_groups,
            )
        else:
            raise NotImplementedError()
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=1,
            batch_first=False,
            gru_skip_op=skip_op,
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        self.lsnr_fc = nn.Sequential(nn.Linear(self.emb_out_dim, 1), nn.Sigmoid())
        self.lsnr_scale = p.lsnr_max - p.lsnr_min
        self.lsnr_offset = p.lsnr_min

    def forward(
        self, feat_erb: Tensor, feat_spec: Tensor, hidden: Tensor
    ) -> Tuple[Tensor, Tensor, Tensor, Tensor, Tensor, Tensor, Tensor]:
        # Encodes erb; erb should be in dB scale + normalized; Fe are number of erb bands.
        # erb: [B, 1, T, Fe]
        # spec: [B, 2, T, Fc]
        # b, _, t, _ = feat_erb.shape

        # feat erb branch
        e0 = self.erb_conv0(feat_erb)  # [B, C, T, F]
        e1 = self.erb_conv1(e0)  # [B, C*2, T, F/2]
        e2 = self.erb_conv2(e1)  # [B, C*4, T, F/4]
        e3 = self.erb_conv3(e2)  # [B, C*4, T, F/4]
        emb = e3.permute(0, 2, 3, 1).flatten(2, 3)  # [B, T, C * F]

        # feat spec branch
        c0 = self.df_conv0(feat_spec)  # [B, C, T, Fc]
        c1 = self.df_conv1(c0)  # [B, C*2, T, Fc/2]
        cemb = c1.permute(0, 2, 3, 1)  # [B, T, -1]
        cemb = self.df_fc_emb(cemb)  # [T, B, C * F/4]

        # combine
        emb = self.combine(emb, cemb)
        emb, hidden = self.emb_gru(emb, hidden)  # [B, T, -1]
        return e0, e1, e2, e3, emb, c0, hidden


class ErbDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        assert p.nb_erb % 8 == 0, "erb_bins should be divisible by 8"

        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.emb_hidden_dim
        self.emb_out_dim = p.conv_ch * p.nb_erb // 4

        if p.emb_gru_skip == "none":
            skip_op = None
        elif p.emb_gru_skip == "identity":
            assert self.emb_in_dim == self.emb_out_dim, "Dimensions do not match"
            skip_op = partial(nn.Identity)
        elif p.emb_gru_skip == "groupedlinear":
            skip_op = partial(
                GroupedLinearEinsum,
                input_size=self.emb_in_dim,
                hidden_size=self.emb_out_dim,
                groups=p.lin_groups,
            )
        else:
            raise NotImplementedError()
        self.emb_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            output_size=self.emb_out_dim,
            num_layers=p.emb_num_layers - 1,
            batch_first=False,
            gru_skip_op=skip_op,
            linear_groups=p.lin_groups,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        tconv_layer = partial(
            ConvTranspose2dNormAct,
            kernel_size=p.convt_kernel,
            bias=False,
            separable=True,
        )
        conv_layer = partial(
            Conv2dNormAct,
            bias=False,
            separable=True,
        )
        # convt: TransposedConvolution, convp: Pathway (encoder to decoder) convolutions
        self.conv3p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt3 = conv_layer(p.conv_ch, p.conv_ch, kernel_size=p.conv_kernel)
        self.conv2p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt2 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv1p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.convt1 = tconv_layer(p.conv_ch, p.conv_ch, fstride=2)
        self.conv0p = conv_layer(p.conv_ch, p.conv_ch, kernel_size=1)
        self.conv0_out = conv_layer(
            p.conv_ch, 1, kernel_size=p.conv_kernel, activation_layer=nn.Sigmoid
        )

    def forward(
        self,
        emb: Tensor,
        e3: Tensor,
        e2: Tensor,
        e1: Tensor,
        e0: Tensor,
        hidden: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        # Estimates erb mask
        b, _, t, f8 = e3.shape
        emb, hidden = self.emb_gru(emb, hidden)
        emb = emb.view(1, 1, f8, -1).permute(0, 3, 1, 2)  # [B, C*8, T, F/8]
        e3 = self.convt3(self.conv3p(e3) + emb)  # [B, C*4, T, F/4]
        e2 = self.convt2(self.conv2p(e2) + e3)  # [B, C*2, T, F/2]
        e1 = self.convt1(self.conv1p(e1) + e2)  # [B, C, T, F]
        m = self.conv0_out(self.conv0p(e0) + e1)  # [B, 1, T, F]
        return m, hidden


class DfDecoder(nn.Module):
    def __init__(self):
        super().__init__()
        p = ModelParams()
        layer_width = p.conv_ch

        self.emb_in_dim = p.conv_ch * p.nb_erb // 4
        self.emb_dim = p.df_hidden_dim

        self.df_n_hidden = p.df_hidden_dim
        self.df_n_layers = p.df_num_layers
        self.df_order = p.df_order
        self.df_bins = p.nb_df
        self.df_out_ch = p.df_order * 2

        conv_layer = partial(Conv2dNormAct, separable=True, bias=False)
        kt = p.df_pathway_kernel_size_t
        self.conv_buffer_size = kt - 1
        self.df_convp = conv_layer(
            layer_width, self.df_out_ch, fstride=1, kernel_size=(kt, 1)
        )

        self.df_gru = SqueezedGRU_S(
            self.emb_in_dim,
            self.emb_dim,
            num_layers=self.df_n_layers,
            batch_first=True,
            gru_skip_op=None,
            linear_act_layer=partial(nn.ReLU, inplace=True),
        )
        p.df_gru_skip = p.df_gru_skip.lower()
        assert p.df_gru_skip in ("none", "identity", "groupedlinear")
        self.df_skip: Optional[nn.Module]
        if p.df_gru_skip == "none":
            self.df_skip = None
        elif p.df_gru_skip == "identity":
            assert p.emb_hidden_dim == p.df_hidden_dim, "Dimensions do not match"
            self.df_skip = nn.Identity()
        elif p.df_gru_skip == "groupedlinear":
            self.df_skip = GroupedLinearEinsum(
                self.emb_in_dim, self.emb_dim, groups=p.lin_groups
            )
        else:
            raise NotImplementedError()
        self.df_out: nn.Module
        out_dim = self.df_bins * self.df_out_ch
        df_out = GroupedLinearEinsum(
            self.df_n_hidden, out_dim, groups=p.lin_groups, activation=nn.Tanh()
        )
        self.df_out = nn.Sequential(df_out)
        self.df_fc_a = nn.Sequential(nn.Linear(self.df_n_hidden, 1), nn.Sigmoid())

    def forward(self, emb: Tensor, c0: Tensor, hidden: Tensor) -> Tuple[Tensor, Tensor]:
        b, t, _ = emb.shape
        c, hidden = self.df_gru(emb, hidden)  # [B, T, H], H: df_n_hidden
        if self.df_skip is not None:
            c = c + self.df_skip(emb)
        c0 = self.df_convp(c0).permute(0, 2, 3, 1)  # [B, T, F, O*2], channels_last
        c = self.df_out(c)  # [B, T, F*O*2], O: df_order
        c = c.view(b, t, self.df_bins, self.df_out_ch) + c0  # [B, T, F, O*2]
        return c, hidden


class ExportableStreamingMinimalTorchDF(nn.Module):
    def __init__(
        self,
        fft_size,
        hop_size,
        nb_bands,
        enc,
        df_dec,
        erb_dec,
        erb_indices,
        df_order=5,
        lookahead=2,
        conv_lookahead=2,
        nb_df=96,
        alpha=0.99,
        min_db_thresh=-10.0,
        max_db_erb_thresh=30.0,
        max_db_df_thresh=20.0,
        normalize_atten_lim=20.0,
        silence_thresh=1e-7,
        sr=48000,
    ):
        # All complex numbers are stored as floats for ONNX compatibility
        super().__init__()

        self.fft_size = torch.tensor(fft_size, dtype=torch.float32)
        self.frame_size = hop_size  # dimension "f" in Float[f]
        self.window_size = fft_size
        self.window_size_h = fft_size // 2
        self.freq_size = fft_size // 2 + 1  # dimension "F" in Float[F]
        self.wnorm = 1.0 / (self.window_size**2 / (2 * self.frame_size))
        self.df_order = df_order
        self.lookahead = lookahead
        self.sr = sr

        # Initialize the vorbis window: sin(pi/2*sin^2(pi*n/N))
        window = torch.sin(
            0.5 * torch.pi * (torch.arange(self.fft_size) + 0.5) / self.window_size_h
        )
        window = torch.sin(
            0.5 * torch.pi * window**2,
        )
        self.register_buffer("window", window)

        self.nb_df = nb_df
        self.erb_indices = torch.from_numpy(erb_indices.astype(np.int64))
        self.nb_bands = nb_bands

        self.register_buffer(
            "forward_erb_matrix",
            self.erb_fb(self.erb_indices, normalized=True, inverse=False),
        )
        self.register_buffer(
            "inverse_erb_matrix",
            self.erb_fb(self.erb_indices, normalized=True, inverse=True),
        )

        ### Model
        self.enc = Encoder()
        self.enc.load_state_dict(enc.state_dict())
        self.enc.eval()

        # Instead of padding we put tensor with buffers into encoder
        # I didn't checked receptived fields of convolution, but equallity tests are working
        self.enc.erb_conv0 = self.remove_conv_block_padding(self.enc.erb_conv0)
        self.enc.df_conv0 = self.remove_conv_block_padding(self.enc.df_conv0)

        # Instead of padding we put tensor with buffers into df_decoder
        self.df_dec = DfDecoder()
        self.df_dec.load_state_dict(df_dec.state_dict())
        self.df_dec.eval()
        self.df_dec.df_convp = self.remove_conv_block_padding(self.df_dec.df_convp)

        self.erb_dec = ErbDecoder()
        self.erb_dec.load_state_dict(erb_dec.state_dict())
        self.erb_dec.eval()
        ### End Model

        self.alpha = alpha

        # RFFT
        # FFT operations are performed as matmuls for ONNX compatability
        self.register_buffer(
            "rfft_matrix",
            torch.view_as_real(torch.fft.rfft(torch.eye(self.window_size))).transpose(
                0, 1
            ),
        )
        self.register_buffer("irfft_matrix", torch.linalg.pinv(self.rfft_matrix))

        # Thresholds
        self.register_buffer("min_db_thresh", torch.tensor([min_db_thresh]))
        self.register_buffer("max_db_erb_thresh", torch.tensor([max_db_erb_thresh]))
        self.register_buffer("max_db_df_thresh", torch.tensor([max_db_df_thresh]))
        self.normalize_atten_lim = torch.tensor(normalize_atten_lim)
        self.silence_thresh = torch.tensor(silence_thresh)
        self.linspace_erb = [-60.0, -90.0]
        self.linspace_df = [0.001, 0.0001]

        self.erb_norm_state_shape = (self.nb_bands,)
        self.band_unit_norm_state_shape = (
            1,
            self.nb_df,
            1,
        )  # [bs=1, nb_df, mean of complex value = 1]
        self.analysis_mem_shape = (self.frame_size,)
        self.synthesis_mem_shape = (self.frame_size,)
        self.rolling_erb_buf_shape = (
            1,
            1,
            conv_lookahead + 1,
            self.nb_bands,
        )  # [B, 1, conv kernel size, nb_bands]
        self.rolling_feat_spec_buf_shape = (
            1,
            2,
            conv_lookahead + 1,
            self.nb_df,
        )  # [B, 2 - complex, conv kernel size, nb_df]
        self.rolling_c0_buf_shape = (
            1,
            self.enc.df_conv0_ch,
            self.df_order,
            self.nb_df,
        )  # [B, conv hidden, df_order, nb_df]
        self.rolling_spec_buf_x_shape = (
            max(self.df_order, conv_lookahead),
            self.freq_size,
            2,
        )  # [number of specs to save, ...]
        self.rolling_spec_buf_y_shape = (
            self.df_order + conv_lookahead,
            self.freq_size,
            2,
        )  # [number of specs to save, ...]
        self.enc_hidden_shape = (
            1,
            1,
            self.enc.emb_dim,
        )  # [n_layers=1, batch_size=1, emb_dim]
        self.erb_dec_hidden_shape = (
            2,
            1,
            self.erb_dec.emb_dim,
        )  # [n_layers=2, batch_size=1, emb_dim]
        self.df_dec_hidden_shape = (
            2,
            1,
            self.df_dec.emb_dim,
        )  # [n_layers=2, batch_size=1, emb_dim]

        # States
        state_shapes = [
            self.erb_norm_state_shape,
            self.band_unit_norm_state_shape,
            self.analysis_mem_shape,
            self.synthesis_mem_shape,
            self.rolling_erb_buf_shape,
            self.rolling_feat_spec_buf_shape,
            self.rolling_c0_buf_shape,
            self.rolling_spec_buf_x_shape,
            self.rolling_spec_buf_y_shape,
            self.enc_hidden_shape,
            self.erb_dec_hidden_shape,
            self.df_dec_hidden_shape,
        ]
        self.state_lens = [math.prod(x) for x in state_shapes]
        self.states_full_len = sum(self.state_lens)

        # Zero buffers
        self.register_buffer("zero_gains", torch.zeros(self.nb_bands))
        self.register_buffer(
            "zero_coefs", torch.zeros(self.rolling_c0_buf_shape[2], self.nb_df, 2)
        )

    @staticmethod
    def remove_conv_block_padding(original_conv: nn.Module) -> nn.Module:
        """
        Remove paddings for convolutions in the original model

        Parameters:
            original_conv:  nn.Module - original convolution module

        Returns:
            output:         nn.Module - new convolution module without paddings
        """
        new_modules = []

        for module in original_conv:
            if not isinstance(module, nn.ConstantPad2d):
                new_modules.append(module)

        return nn.Sequential(*new_modules)

    def erb_fb(
        self, widths: Tensor, normalized: bool = True, inverse: bool = False
    ) -> Tensor:
        """
        Generate the erb filterbank
        Taken from https://github.com/Rikorose/DeepFilterNet/blob/fa926662facea33657c255fd1f3a083ddc696220/DeepFilterNet/df/modules.py#L206
        Numpy removed from original code

        Parameters:
            widths:     Tensor - widths of the erb bands
            normalized: bool - normalize to constant energy per band
            inverse:    bool - inverse erb filterbank

        Returns:
            fb:         Tensor - erb filterbank
        """
        n_freqs = int(torch.sum(widths))
        all_freqs = torch.linspace(0, self.sr // 2, n_freqs + 1)[:-1]

        b_pts = torch.cumsum(
            torch.cat([torch.tensor([0]), widths]), dtype=torch.int32, dim=0
        )[:-1]

        fb = torch.zeros((all_freqs.shape[0], b_pts.shape[0]))
        for i, (b, w) in enumerate(zip(b_pts.tolist(), widths.tolist())):
            fb[b : b + w, i] = 1

        # Normalize to constant energy per resulting band
        if inverse:
            fb = fb.t()
            if not normalized:
                fb /= fb.sum(dim=1, keepdim=True)
        else:
            if normalized:
                fb /= fb.sum(dim=0)

        return fb

    @staticmethod
    def mul_complex(t1, t2):
        """
        Compute multiplication of two complex numbers in view_as_real format.

        Parameters:
            t1:         Float[F, 2] - First number
            t2:         Float[F, 2] - Second number

        Returns:
            output:     Float[F, 2] - final multiplication of two complex numbers
        """
        # if not torch.onnx.is_in_onnx_export():
        t1_real = t1[..., 0]
        t1_imag = t1[..., 1]
        t2_real = t2[..., 0]
        t2_imag = t2[..., 1]
        return torch.stack(
            (
                t1_real * t2_real - t1_imag * t2_imag,
                t1_real * t2_imag + t1_imag * t2_real,
            ),
            dim=-1,
        )
        # return t1 * t2
        # return OnnxComplexMul.apply(t1, t2)

    def erb(self, input_data: Tensor, erb_eps: float = 1e-10) -> Tensor:
        """
        Original code - pyDF/src/lib.rs - erb()
        Calculating ERB features for each frame.

        Parameters:
            input_data:     Float[T, F] or Float[F] - audio spectrogram

        Returns:
            erb_features:   Float[T, ERB] or Float[ERB] - erb features for given spectrogram
        """

        magnitude_squared = torch.sum(input_data**2, dim=-1)
        erb_features = magnitude_squared.matmul(self.forward_erb_matrix)
        erb_features_db = 10.0 * torch.log10(erb_features + erb_eps)

        return erb_features_db

    @staticmethod
    def band_mean_norm_erb(
        xs: Tensor, erb_norm_state: Tensor, alpha: float, denominator: float = 40.0
    ) -> Tuple[Tensor, Tensor]:
        """
        Original code - libDF/src/lib.rs - band_mean_norm()
        Normalizing ERB features. And updates the normalization state.

        Parameters:
            xs:             Float[ERB] - erb features
            erb_norm_state: Float[ERB] - normalization state from previous step
            alpha:          float - alpha value which is needed for adaptation of the normalization state for given scale.
            denominator:    float - denominator for normalization

        Returns:
            output:         Float[ERB] - normalized erb features
            erb_norm_state: Float[ERB] - updated normalization state
        """
        new_erb_norm_state = xs * (1 - alpha) + erb_norm_state * alpha
        output = (xs - new_erb_norm_state) / denominator

        return output, new_erb_norm_state

    @staticmethod
    def band_unit_norm(
        xs: Tensor, band_unit_norm_state, alpha: float
    ) -> Tuple[Tensor, Tensor]:
        """
        Original code - libDF/src/lib.rs - band_unit_norm()
        Normalizing Deep Filtering features. And updates the normalization state.

        Parameters:
            xs:                     Float[1, DF, 2] - deep filtering features
            band_unit_norm_state:   Float[1, DF, 1] - normalization state from previous step
            alpha:                  float - alpha value which is needed for adaptation of the normalization state for given scale.

        Returns:
            output:                 Float[1, DF] - normalized deep filtering features
            band_unit_norm_state:   Float[1, DF, 1] - updated normalization state
        """
        xs_abs = torch.linalg.norm(xs, dim=-1, keepdim=True)  # xs.abs() from complexxs
        new_band_unit_norm_state = xs_abs * (1 - alpha) + band_unit_norm_state * alpha
        output = xs / new_band_unit_norm_state.sqrt()

        return output, new_band_unit_norm_state

    def frame_analysis(
        self, input_frame: Tensor, analysis_mem: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Original code - libDF/src/lib.rs - frame_analysis()
        Calculating spectrograme for one frame. Every frame is concated with buffer from previous frame.

        Parameters:
            input_frame:    Float[f] - Input raw audio frame
            analysis_mem:   Float[f] - Previous frame

        Returns:
            output:         Float[F, 2] - Spectrogram
            analysis_mem:   Float[f] - Saving current frame for next iteration
        """
        # First part of the window on the previous frame
        # Second part of the window on the new input frame
        buf = torch.cat([analysis_mem, input_frame]) * self.window
        # rfft_buf = torch.matmul(buf, self.rfft_matrix) * self.wnorm
        rfft_buf = torch.view_as_real(torch.fft.rfft(buf)) * self.wnorm

        # Copy input to analysis_mem for next iteration
        return rfft_buf, input_frame

    def frame_synthesis(
        self, x: Tensor, synthesis_mem: Tensor
    ) -> Tuple[Tensor, Tensor]:
        """
        Original code - libDF/src/lib.rs - frame_synthesis()
        Inverse rfft for one frame. Every frame is summarized with buffer from previous frame.
        And saving buffer for next frame.

        Parameters:
            x:     Float[F, 2] - Enhanced audio spectrogram
            synthesis_mem:  Float[f] - Previous synthesis frame

        Returns:
            output:         Float[f] - Enhanced audio
            synthesis_mem:  Float[f] - Saving current frame
        """
        # x - [F=481, 2]
        # self.irfft_matrix - [fft_size=481, 2, f=960]
        # [f=960]
        x = (
            torch.einsum("fi,fij->j", x, self.irfft_matrix)
            * self.fft_size
            * self.window
        )
        # x = torch.cat([x[:, 0], torch.zeros(479)])
        # x = torch.fft.irfft(torch.view_as_complex(x)) * self.fft_size * self.window

        x_first, x_second = torch.split(
            x, [self.frame_size, self.window_size - self.frame_size]
        )

        output = x_first + synthesis_mem

        return output, x_second.view(self.window_size - self.frame_size)

    def apply_mask(self, spec: Tensor, gains: Tensor) -> Tensor:
        """
        Original code - libDF/src/lib.rs - apply_interp_band_gain()

        Applying ERB Gains for input spectrogram

        Parameters:
            spec:   Float[F, 2] - Input frame spectrogram
            gains:  Float[ERB] - ERB gains from erb decoder

        Returns:
            spec:   Float[F] - Spectrogram with applyed ERB gains
        """
        gains = gains.matmul(self.inverse_erb_matrix)
        spec = spec * gains.unsqueeze(-1)

        return spec

    def deep_filter(
        self, gain_spec: Tensor, coefs: Tensor, rolling_spec_buf_x: Tensor
    ) -> Tensor:
        """
        Original code - libDF/src/tract.rs - df()

        Applying Deep Filtering to gained spectrogram by multiplying coefs to rolling_buffer_x (spectrograms from past / future).
        Deep Filtering replacing lower self.nb_df spec bands.

        Parameters:
            gain_spec:              Float[F, 2] - spectrogram after ERB gains applied
            coefs:                  Float[DF, BUF, 2] - coefficients for deep filtering from df decoder
            rolling_spec_buf_x:     Float[buffer_size, F, 2] - spectrograms from past / future

        Returns:
            gain_spec:              Float[F, 2] - spectrogram after deep filtering
        """
        stacked_input_specs = rolling_spec_buf_x[:, : self.nb_df]
        mult = self.mul_complex(stacked_input_specs, coefs)
        gain_spec[: self.nb_df] = torch.sum(mult, dim=0)
        return gain_spec

    def forward(
        self,
        input_frame: Tensor,
        erb_norm_state: Tensor,
        band_unit_norm_state: Tensor,
        analysis_mem: Tensor,
        synthesis_mem: Tensor,
        rolling_erb_buf: Tensor,
        rolling_feat_spec_buf: Tensor,
        rolling_c0_buf: Tensor,
        rolling_spec_buf_x: Tensor,
        rolling_spec_buf_y: Tensor,
        enc_hidden: Tensor,
        erb_dec_hidden: Tensor,
        df_dec_hidden: Tensor,
    ) -> Tuple[
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
        Tensor,
    ]:
        """
        Enhancing input audio frame

        Parameters:
            input_frame:        Float[t] - Input raw audio frame

        Returns:
            enhanced_frame:     Float[t] - Enhanced audio frame
        """
        assert input_frame.ndim == 1, "only bs=1 and t=frame_size supported"
        assert (
            input_frame.shape[0] == self.frame_size
        ), "input_frame must be bs=1 and t=frame_size"

        spectrogram, new_analysis_mem = self.frame_analysis(input_frame, analysis_mem)
        spectrogram = spectrogram.unsqueeze(
            0
        )  # [1, freq_size, 2] reshape needed for easier stacking buffers
        new_rolling_spec_buf_x = torch.cat(
            [rolling_spec_buf_x[1:, ...], spectrogram]
        )  # [n_frames=5, 481, 2]
        # rolling_spec_buf_y - [n_frames=7, 481, 2] n_frames=7 for compatability with original code, but in code we use only one frame
        new_rolling_spec_buf_y = torch.cat([rolling_spec_buf_y[1:, ...], spectrogram])

        erb_feat, new_erb_norm_state = self.band_mean_norm_erb(
            self.erb(spectrogram).squeeze(0), erb_norm_state, alpha=self.alpha
        )  # [ERB]
        spec_feat, new_band_unit_norm_state = self.band_unit_norm(
            spectrogram[:, : self.nb_df], band_unit_norm_state, alpha=self.alpha
        )  # [1, DF, 2]

        erb_feat = erb_feat[
            None, None, None, ...
        ]  # [b=1, conv_input_dim=1, t=1, n_erb=32]
        spec_feat = spec_feat[None, ...].permute(
            0, 3, 1, 2
        )  # [bs=1, conv_input_dim=2, t=1, df_order=96]

        # (1, 1, T, self.nb_bands)
        new_rolling_erb_buf = torch.cat([rolling_erb_buf[:, :, 1:, :], erb_feat], dim=2)

        #  (1, 2, T, self.nb_df)
        new_rolling_feat_spec_buf = torch.cat(
            [rolling_feat_spec_buf[:, :, 1:, :], spec_feat], dim=2
        )

        e0, e1, e2, e3, emb, c0, new_enc_hidden = self.enc(
            new_rolling_erb_buf, new_rolling_feat_spec_buf, enc_hidden
        )

        # erb_dec
        # [BS=1, 1, T=1, ERB]
        new_gains, new_erb_dec_hidden = self.erb_dec(
            emb, e3, e2, e1, e0, erb_dec_hidden
        )
        gains = new_gains.view(self.nb_bands)

        # df_dec
        new_rolling_c0_buf = torch.cat([rolling_c0_buf[:, :, 1:, :], c0], dim=2)
        # new_coefs - [BS=1, T=1, F, O*2]
        new_coefs, new_df_dec_hidden = self.df_dec(
            emb, new_rolling_c0_buf, df_dec_hidden
        )
        coefs = new_coefs.view(self.nb_df, -1, 2).permute(1, 0, 2)

        # Applying features
        current_spec = new_rolling_spec_buf_y[self.df_order - 1]
        current_spec = self.apply_mask(current_spec.clone(), gains)
        current_spec = self.deep_filter(
            current_spec.clone(), coefs, new_rolling_spec_buf_x
        )

        enhanced_audio_frame, new_synthesis_mem = self.frame_synthesis(
            current_spec, synthesis_mem
        )

        return (
            enhanced_audio_frame,
            new_erb_norm_state,
            new_band_unit_norm_state,
            new_analysis_mem,
            new_synthesis_mem,
            new_rolling_erb_buf,
            new_rolling_feat_spec_buf,
            new_rolling_c0_buf,
            new_rolling_spec_buf_x,
            new_rolling_spec_buf_y,
            new_enc_hidden,
            new_erb_dec_hidden,
            new_df_dec_hidden,
        )


class TorchDFMinimalPipeline(nn.Module):
    def __init__(
        self,
        model_base_dir="DeepFilterNet3",
        epoch="best",
        device="cpu",
    ):
        super().__init__()

        model, state, _ = init_df(
            config_allow_defaults=True,
            model_base_dir=model_base_dir,
            epoch=epoch,
        )
        model.eval()
        p = ModelParams()

        self.hop_size = p.hop_size
        self.fft_size = p.fft_size
        self.sample_rate = p.sr
        
        self.torch_streaming_model = ExportableStreamingMinimalTorchDF(
            nb_bands=p.nb_erb,
            hop_size=p.hop_size,
            fft_size=p.fft_size,
            enc=model.enc,
            df_dec=model.df_dec,
            erb_dec=model.erb_dec,
            df_order=p.df_order,
            conv_lookahead=p.conv_lookahead,
            nb_df=p.nb_df,
            sr=self.sample_rate,
            erb_indices=state.erb_widths()
        )
        self.torch_streaming_model = self.torch_streaming_model.to(device)

        analysis_mem = torch.zeros(self.torch_streaming_model.analysis_mem_shape)
        synthesis_mem = torch.zeros(self.torch_streaming_model.synthesis_mem_shape)
        rolling_erb_buf = torch.zeros(self.torch_streaming_model.rolling_erb_buf_shape)
        rolling_feat_spec_buf = torch.zeros(
            self.torch_streaming_model.rolling_feat_spec_buf_shape
        )
        rolling_c0_buf = torch.zeros(self.torch_streaming_model.rolling_c0_buf_shape)
        rolling_spec_buf_x = torch.zeros(
            self.torch_streaming_model.rolling_spec_buf_x_shape
        )
        rolling_spec_buf_y = torch.zeros(
            self.torch_streaming_model.rolling_spec_buf_y_shape
        )
        enc_hidden = torch.zeros(self.torch_streaming_model.enc_hidden_shape)
        erb_dec_hidden = torch.zeros(self.torch_streaming_model.erb_dec_hidden_shape)
        df_dec_hidden = torch.zeros(self.torch_streaming_model.df_dec_hidden_shape)

        erb_norm_state = (
            torch.linspace(
                self.torch_streaming_model.linspace_erb[0],
                self.torch_streaming_model.linspace_erb[1],
                self.torch_streaming_model.nb_bands,
            )
            .view(self.torch_streaming_model.erb_norm_state_shape)
            .to(torch.float32)
        )  # float() to fix export issue

        band_unit_norm_state = (
            torch.linspace(
                self.torch_streaming_model.linspace_df[0],
                self.torch_streaming_model.linspace_df[1],
                self.torch_streaming_model.nb_df,
            )
            .view(self.torch_streaming_model.band_unit_norm_state_shape)
            .to(torch.float32)
        )  # float() to fix export issue

        self.states = [
            erb_norm_state,
            band_unit_norm_state,
            analysis_mem,
            synthesis_mem,
            rolling_erb_buf,
            rolling_feat_spec_buf,
            rolling_c0_buf,
            rolling_spec_buf_x,
            rolling_spec_buf_y,
            enc_hidden,
            erb_dec_hidden,
            df_dec_hidden,
        ]
        self.input_names = [
            "input_frame",
            "erb_norm_state",
            "band_unit_norm_state",
            "analysis_mem",
            "synthesis_mem",
            "rolling_erb_buf",
            "rolling_feat_spec_buf",
            "rolling_c0_buf",
            "rolling_spec_buf_x",
            "rolling_spec_buf_y",
            "enc_hidden",
            "erb_dec_hidden",
            "df_dec_hidden",
        ]
        self.output_names = [
            "enhanced_audio_frame",
            "new_erb_norm_state",
            "new_band_unit_norm_state",
            "new_analysis_mem",
            "new_synthesis_mem",
            "new_rolling_erb_buf",
            "new_rolling_feat_spec_buf",
            "new_rolling_c0_buf",
            "new_rolling_spec_buf_x",
            "new_rolling_spec_buf_y",
            "new_enc_hidden",
            "new_erb_dec_hidden",
            "new_df_dec_hidden",
        ]

    def forward(self, input_audio: Tensor, sample_rate: int) -> Tensor:
        """
        Denoising audio frame using exportable fully torch model.

        Parameters:
            input_audio:      Float[1, t] - Input audio
            sample_rate:      Int - Sample rate

        Returns:
            enhanced_audio:   Float[1, t] - Enhanced input audio
        """
        assert (
            input_audio.shape[0] == 1
        ), f"Only mono supported! Got wrong shape! {input_audio.shape}"
        assert (
            sample_rate == self.sample_rate
        ), f"Only {self.sample_rate} supported! Got wrong sample rate! {sample_rate}"

        input_audio = input_audio.squeeze(0)
        orig_len = input_audio.shape[0]

        # padding taken from
        # https://github.com/Rikorose/DeepFilterNet/blob/fa926662facea33657c255fd1f3a083ddc696220/DeepFilterNet/df/enhance.py#L229
        hop_size_divisible_padding_size = (
            self.hop_size - orig_len % self.hop_size
        ) % self.hop_size
        orig_len += hop_size_divisible_padding_size
        input_audio = F.pad(
            input_audio, (0, self.fft_size + hop_size_divisible_padding_size)
        )

        chunked_audio = torch.split(input_audio, self.hop_size)

        output_frames = []

        for input_frame in chunked_audio:
            enhanced_audio_frame, *self.states = self.torch_streaming_model(
                input_frame, *self.states
            )

            output_frames.append(enhanced_audio_frame)

        enhanced_audio = torch.cat(output_frames).unsqueeze(
            0
        )  # [t] -> [1, t] typical mono format

        # taken from
        # https://github.com/Rikorose/DeepFilterNet/blob/fa926662facea33657c255fd1f3a083ddc696220/DeepFilterNet/df/enhance.py#L248
        d = self.fft_size - self.hop_size
        enhanced_audio = enhanced_audio[:, d : orig_len + d]

        return enhanced_audio


def main(args):
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    torch_df = TorchDFMinimalPipeline(device=args.device)

    # torchaudio normalize=True, fp32 return
    noisy_audio, sr = torchaudio.load(args.audio_path, channels_first=True)
    noisy_audio = noisy_audio.mean(dim=0).unsqueeze(0).to(args.device)  # stereo to mono

    enhanced_audio = torch_df(noisy_audio, sr).detach().cpu()

    torchaudio.save(
        args.output_path, enhanced_audio, sr, encoding="PCM_S", bits_per_sample=16
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Denoising one audio with DF3 model using torch only"
    )
    parser.add_argument(
        "--audio-path", type=str, required=True, help="Path to audio file"
    )
    parser.add_argument(
        "--output-path", type=str, required=True, help="Path to output file"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        choices=["cuda", "cpu"],
        help="Device to run on",
    )

    main(parser.parse_args())
