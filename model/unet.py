import torch
import torch.nn as nn
from typing import List, Optional, Tuple, Union

class DownSample1D(nn.Module):
    """
    1D-UNET downsampling block.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
    ):
        """
        Args:
            output_channels: number of output channels.
            activation_fn: name of the activation function to use.
                Should be one of "gelu",
                "gelu-no-approx", "relu", "swish", "silu", "sin".
            num_layers: number of convolution layers.
            name: module name.
        """
        
        super().__init__()
        self.first_layer = [nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )]
        

        self.next_layers = [
            nn.Conv1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )
            for _ in range(num_layers-1)
        ]
        self.conv_layers = nn.ModuleList(self.first_layer + self.next_layers)

        self.avg_pool = nn.AvgPool1d(
            kernel_size=2,
            stride=2,
            padding=0,
        )
        self.activation_fn = nn.SiLU()


    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, conv_layer in enumerate(self.conv_layers):
            x = self.activation_fn(conv_layer(x))

        hidden = x
        x = self.avg_pool(hidden)
        return x, hidden
    


class UpSample1D(nn.Module):
    """
    1D-UNET upsampling block.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
    ):
        """
        Args:
            output_channels: number of output channels.
            activation_fn: name of the activation function to use.
                Should be one of "gelu",
                "gelu-no-approx", "relu", "swish", "silu", "sin".
            interpolation_method: Method to be used for upsampling interpolation.
                Should be one of "nearest", "linear", "cubic", "lanczos3", "lanczos5".
            num_layers: number of convolution layers.
            name: module name.
        """
        super().__init__()

        self._first_layer = [nn.ConvTranspose1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )]


        self._next_layers = [
            nn.ConvTranspose1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                padding=1,
            )
            for _ in range(num_layers-1)
        ]

        self.conv_layers = nn.ModuleList(self._first_layer + self._next_layers)

        self._activation_fn = nn.SiLU()



    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        for i, conv_layer in enumerate(self.conv_layers):
            x = self._activation_fn(conv_layer(x))      

        # Different order than in Haiku because the channels are changed when going 
        # from Haiku to Torch.
        x = nn.functional.interpolate(x, size=2 * x.shape[2], mode="nearest")


        return x



class FinalConv1D(nn.Module):
    """
    Final output block of the 1D-UNET.
    """

    def __init__(
        self,
        input_channels: int,
        output_channels: int,
        num_layers: int = 2,
    ):
        """
        Args:
            output_channels: number of output channels.
            activation_fn: name of the activation function to use.
                Should be one of "gelu",
                "gelu-no-approx", "relu", "swish", "silu", "sin".
            num_layers: number of convolution layers.
            name: module name.
        """
        super().__init__()

        self._first_layer = [nn.Conv1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )]

        self._next_layers = [
            nn.Conv1d(
                in_channels=output_channels,
                out_channels=output_channels,
                kernel_size=3,
                stride=1,
                dilation=1,
                padding="same",
            )
            for _ in range(num_layers-1)
        ]
        self.conv_layers = nn.ModuleList(self._first_layer + self._next_layers)

        self._activation_fn = nn.SiLU()



    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, conv_layer in enumerate(self.conv_layers):
            x = conv_layer(x)
            if i < len(self.conv_layers) - 1:
                x = self._activation_fn(x)
        return x


class UNET1DSegmentationHead(nn.Module):
    """
    1D-UNET based head to be plugged on top of a pretrained model to perform
    semantic segmentation.
    """

    def __init__(
        self,
        embed_dim: int = 4,
        num_classes: int = 2,
        seq_length: int = 512,
        output_channels_list: Tuple[int, ...] = (64, 128, 256),
        num_conv_layers_per_block: int = 2,
    ):
        """
        Args:
            num_classes: number of classes to segment
            output_channels_list: list of the number of output channel at each level of
                the UNET
            num_conv_layers_per_block: number of convolution layers per block.
        """
        super().__init__()
        self._num_pooling_layers = len(output_channels_list)

        downsample_input_channels_list = (embed_dim, ) + output_channels_list[:-1]

        output_channels_list_reversed = tuple(reversed(output_channels_list))
        upsample_input_channels_list = (output_channels_list[-1],) + output_channels_list_reversed
        upsample_output_channels_list =  output_channels_list_reversed

        self._downsample_blocks = nn.ModuleList([
            DownSample1D(
                input_channels= input_channels,
                output_channels=output_channels,
                num_layers=num_conv_layers_per_block,
            )
            for input_channels, output_channels in zip(downsample_input_channels_list, output_channels_list)
        ])

        self._upsample_blocks = nn.ModuleList([
            UpSample1D(
                input_channels = input_channels,
                output_channels=output_channels,
                num_layers=num_conv_layers_per_block,
            )
            for input_channels, output_channels in zip(upsample_input_channels_list, upsample_output_channels_list)
        ])

        self.final_block = FinalConv1D(
            input_channels=output_channels_list[0],
            output_channels=num_classes * 2,
            num_layers=num_conv_layers_per_block,
        )

        self.fc1 = nn.Linear(in_features=embed_dim * seq_length, out_features=seq_length) # for mse computing
        self.fc2 = nn.Linear(in_features=seq_length, out_features=1)

        self.activation_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x (N, Cin, Lin)
        x = torch.transpose(x, 2,1)
        #print(f'x shape: {x.shape}')

        if x.shape[2] % 2**self._num_pooling_layers:
            raise ValueError(
                "Input length must be divisible by the 2 to the power of"
                " number of poolign layers."
            )

        hiddens = []
        for downsample_block in self._downsample_blocks:
            x, hidden = downsample_block(x)
            hiddens.append(hidden)

        for i, (upsample_block, hidden) in enumerate(zip(self._upsample_blocks, reversed(hiddens))):
            x = upsample_block(x) + hidden
        x = self.final_block(x)

        x = self.activation_fn(x)

        x = torch.transpose(x, 2, 1)
        

        x = x.reshape(x.shape[0], -1)

        x = self.activation_fn(self.fc1(x))
        x = self.fc2(x).squeeze()

        #print(f'output x shape:{x.shape}')

        return x
    
    
# For YeastTSSProphet
import torch
import torch.nn as nn
from typing import Tuple

class UNET1DRegressor(nn.Module):
    """
    1D-UNet 末端接全连接层，直接输出 3 维连续值（多输出回归）。
    """
    def __init__(
        self,
        embed_dim: int = 4,                 # 输入通道 (one-hot 4)
        seq_length: int = 504,              # 序列长度
        num_outputs: int = 500,               # <<<<<< 需要的输出维度
        output_channels_list: Tuple[int, ...] = (64, 128, 256),
        num_conv_layers_per_block: int = 2,
    ):
        super().__init__()
        self.seq_length = seq_length
        self._num_pool = len(output_channels_list)

        # ---------- 下采样 ----------
        down_in_channels = (embed_dim,) + output_channels_list[:-1]
        self.down_blocks = nn.ModuleList(
            DownSample1D(cin, cout, num_conv_layers_per_block)
            for cin, cout in zip(down_in_channels, output_channels_list)
        )

        # ---------- 上采样 ----------
        up_in_channels = (output_channels_list[-1],) + tuple(reversed(output_channels_list))
        up_out_channels = tuple(reversed(output_channels_list))
        self.up_blocks = nn.ModuleList(
            UpSample1D(cin, cout, num_conv_layers_per_block)
            for cin, cout in zip(up_in_channels, up_out_channels)
        )

        # ---------- 最后卷积 + FC ----------
        self.final_conv = FinalConv1D(
            input_channels=output_channels_list[0],
            output_channels=embed_dim,                       # 回到 4 通道做展平更直观
            num_layers=num_conv_layers_per_block,
        )

        flat_dim = embed_dim * seq_length
        self.fc = nn.Sequential(
            nn.SiLU(),
            nn.Linear(flat_dim, seq_length),
            nn.SiLU(),
            nn.Linear(seq_length, num_outputs)               # <<<<<< 输出 3 维
        )

        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        输入 x: [B, C=(L, 4)]  ->  输出: [B, 3]
        """
        x = x.permute(0,2,1)
        # Torch 版本与之前一致：Conv1d 期望 [B, C, L]
        if x.shape[2] % 2 ** self._num_pool:
            raise ValueError(f"Sequence length {x.shape[2]} 不能被 2**num_pool 整除")

        # --------- Down path ---------
        skips = []
        for block in self.down_blocks:
            x, h = block(x)
            skips.append(h)

        # --------- Up path ---------
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x) + skip                      # 跨层连接

        # --------- Final conv & flatten ---------
        x = self.final_conv(x)                       # [B, C(=4), L]
        x = x.permute(0, 2, 1).contiguous()          # -> [B, L, 4]
        x = x.view(x.size(0), -1)                    # -> [B, L*4]

        return self.fc(x), self.fc(x).sum(-1,keepdim=True)                           # -> [B, 500]

class UNET1DDistribution(nn.Module):
    """
    1-D UNet → logits over sequence positions  (shape [B, L]).
    用于 KLDivLoss 与目标分布 p_target 训练。
    """

    def __init__(
        self,
        embed_dim: int = 4,
        seq_length: int = 504,
        output_channels_list: Tuple[int, ...] = (64, 128, 256),
        num_conv_layers_per_block: int = 2,
    ):
        super().__init__()
        self.seq_length = seq_length
        self._num_pool = len(output_channels_list)

        # ---------- Down ----------
        down_in_ch = (embed_dim,) + output_channels_list[:-1]
        self.down_blocks = nn.ModuleList([
            DownSample1D(cin, cout, num_conv_layers_per_block)
            for cin, cout in zip(down_in_ch, output_channels_list)
        ])

        # ---------- Up ----------
        up_in_ch  = (output_channels_list[-1],) + tuple(reversed(output_channels_list))
        up_out_ch = tuple(reversed(output_channels_list))
        self.up_blocks = nn.ModuleList([
            UpSample1D(cin, cout, num_conv_layers_per_block)
            for cin, cout in zip(up_in_ch, up_out_ch)
        ])

        # ---------- Final conv ----------
        # 只要 1 个通道；无需 FC、flatten
        self.final_conv = FinalConv1D(
            input_channels=output_channels_list[0],
            output_channels=1,                      # <-- logits 通道
            num_layers=num_conv_layers_per_block
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        参数
        ----
        x : [B, L, 4]  (与之前 UNET1DRegressor 输入保持一致)

        返回
        ----
        logits : [B, L]   （未做 softmax）
        """
        # 把 [B,L,4] → [B,4,L] 适配 Conv1d
        #x = x.transpose(1, 2)

        if x.shape[2] % 2 ** self._num_pool:
            raise ValueError(f"Sequence length {x.shape[2]} 不能被 2**num_pool 整除")

        # Down path
        skips = []
        for block in self.down_blocks:
            x, h = block(x)
            skips.append(h)

        # Up path
        for block, skip in zip(self.up_blocks, reversed(skips)):
            x = block(x) + skip

        # Final conv -> [B,1,L]
        x = self.final_conv(x).squeeze(1)          # -> [B,L]
        return x

