import torch
import torch.utils.checkpoint
from torch import nn
from torch.nn import BCEWithLogitsLoss, CrossEntropyLoss, MSELoss, SiLU
from transformers import AutoTokenizer, AutoModel
from typing import Tuple



class SegmentNT(nn.Module):
    def __init__(self, embed_dim: int =1024, num_layers: int =2):
        super().__init__()

        # self.num_labels = config.num_labels
        # self.config = config
        # self.num_features = len(config.features)
        self.tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/SegmentNT", trust_remote_code=True)
        pretrained_esm = AutoModel.from_pretrained("/root/autodl-tmp/graph_model/yeast_sequence_processing/DNABERT_and_other_pretrain_model/SegmentNT", trust_remote_code=True)
        self.esm = pretrained_esm.esm

        embed_dim = embed_dim
        num_layers = num_layers

        self.unet = UNET1DSegmentationHead(
            embed_dim=embed_dim,
            num_classes=embed_dim // 2,
            output_channels_list=tuple(
                embed_dim * (2**i) for i in range(num_layers)
            ),
        )
        self.fc1 = nn.Linear(in_features=embed_dim*64, out_features=1024) #for cage tpm computing
        self.fc2 = nn.Linear(in_features=1024, out_features=1)
        self.activation_fn = nn.SiLU()

        self.float()

    def forward(
        self,
        tokenized_seqs,
    ):
        r"""
        labels (`torch.LongTensor` of shape `(batch_size,)`, *optional*):
            Labels for computing the sequence classification/regression loss. Indices should be in `[0, ...,
            config.num_labels - 1]`. If `config.num_labels == 1` a regression loss is computed (Mean-Square loss), If
            `config.num_labels > 1` a classification loss is computed (Cross-Entropy).
        """

        attention_mask = tokenized_seqs != self.tokenizer.pad_token_id
        #print(f'attention_mask.shape:{attention_mask}')
        # 将tokenized_seqs进入冻结参数的esm encoder中，获得embedding,然后使它进入后续的unet
        with torch.no_grad():
            #print(self.esm)
            outputs = self.esm(
                tokenized_seqs,
                attention_mask = attention_mask,
                output_hidden_states=True
            )
        #print(f'outputs.shape:{outputs[0].shape}')
        sequence_output = outputs[0]
        # Remove CLS token
        sequence_output = sequence_output[:,1:,:]
        
        # Invert the channels and sequence length channel
        sequence_output = torch.transpose(sequence_output, 2,1)
        #print(f'sequence_output.shape:{sequence_output.shape}')
        x = self.activation_fn(self.unet(sequence_output))
        

        # Invert the channels and sequence length channel
        #x = torch.transpose(x, 2,1)
        x = torch.reshape(x, (x.shape[0], -1))

        logits = self.fc2(self.activation_fn(self.fc1(x)))

        # Final reshape to have logits per nucleotides, per feature
        # logits = torch.reshape(logits, (x.shape[0], x.shape[1] * 6, self.num_features, 2))

        # Add logits to the ESM outputs
        #outputs["logits"] = logits
        #print(f'logits.shape:{logits.dtype}')
        return logits


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
        embed_dim: int,
        num_classes: int,
        output_channels_list: Tuple[int, ...] = (64, 128, 256),#(64, 128, 256)
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.shape[2] % 2**self._num_pooling_layers:
            raise ValueError(
                f"Input length {x.shape[2]} must be divisible by the 2 to the power of"
                f" number of poolign layers {self._num_pooling_layers}."
            )

        hiddens = []
        for downsample_block in self._downsample_blocks:
            x, hidden = downsample_block(x)
            hiddens.append(hidden)
        
        

        for i, (upsample_block, hidden) in enumerate(zip(self._upsample_blocks, reversed(hiddens))):
            x = upsample_block(x) + hidden
        x = self.final_block(x)
        return x
    