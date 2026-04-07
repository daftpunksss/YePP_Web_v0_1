import torch
from torch import nn
import torch.nn.functional as F 

from typing import Type


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(x, **kwargs) + x
    

class ResidualConcat(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, **kwargs):
        return torch.concat([self.fn(x, **kwargs), x], dim=1)
    
class LocalBlock(nn.Module):
    def __init__(self, in_ch, ks, activation, out_ch=None):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.ks = ks
        
        self.block = nn.Sequential(
                       nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.out_ch,
                            kernel_size=self.ks,
                            padding='same',
                            bias=False
                       ),
                       nn.Dropout(0.2),
                       nn.BatchNorm1d(self.out_ch),
                       activation()
        )
        
    def forward(self, x):
        return self.block(x)
    
class SELayer(nn.Module):
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y
    
from tltorch import TRL

class Bilinear(nn.Module):
    """
    Bilinear layer introduces pairwise product to a NN to model possible combinatorial effects.
    This particular implementation attempts to leverage the number of parameters via low-rank tensor decompositions.

    Parameters
    ----------
    n : int
        Number of input features.
    out : int, optional
        Number of output features. If None, assumed to be equal to the number of input features. The default is None.
    rank : float, optional
        Fraction of maximal to rank to be used in tensor decomposition. The default is 0.05.
    bias : bool, optional
        If True, bias is used. The default is False.

    """
    def __init__(self, n: int, out=None, rank=0.05, bias=False):        
        super().__init__()
        if out is None:
            out = (n, )
        self.trl = TRL((n, n), out, bias=bias, rank=rank) # type: ignore
        self.trl.weight = self.trl.weight.normal_(std=0.00075) # type: ignore
    
    def forward(self, x):
        x = x.unsqueeze(dim=-1)
        return self.trl(x @ x.transpose(-1, -2))

class Concater(nn.Module):
    """
    Concatenates an output of some module with its input alongside some dimension.

    Parameters
    ----------
    module : nn.Module
        Module.
    dim : int, optional
        Dimension to concatenate along. The default is -1.

    """
    def __init__(self, module: nn.Module, dim=-1):        
        super().__init__()
        self.mod = module
        self.dim = dim
    
    def forward(self, x):
        return torch.concat((x, self.mod(x)), dim=self.dim)

class SELayerComplex(nn.Module):
    """
    Squeeze-and-Excite layer.

    Parameters
    ----------
    inp : int
        Middle layer size.
    oup : int
        Input and ouput size.
    reduction : int, optional
        Reduction parameter. The default is 4.

    """
    def __init__(self, inp, oup, reduction=4):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
                nn.Linear(oup, int(inp // reduction)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction), int(inp // reduction)),
                Concater(Bilinear(int(inp // reduction), int(inp // reduction // 2), rank=0.5, bias=True)),
                nn.SiLU(),
                nn.Linear(int(inp // reduction) +  int(inp // reduction // 2), oup),
                nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, = x.size()
        y = x.view(b, c, -1).mean(dim=2)
        y = self.fc(y).view(b, c, 1)
        return x * y
    

class EffBlock(nn.Module):
    def __init__(self, 
                 in_ch, 
                 ks, 
                 resize_factor,
                 filter_per_group,
                 activation, 
                 out_ch=None,
                 se_reduction=None,
                 se_type="complex",
                 inner_dim_calculation="out"
                 ):
        super().__init__()
        self.in_ch = in_ch
        self.out_ch = self.in_ch if out_ch is None else out_ch
        self.resize_factor = resize_factor
        self.se_reduction = resize_factor if se_reduction is None else se_reduction
        self.ks = ks
        self.inner_dim_calculation = inner_dim_calculation
        if inner_dim_calculation == "out":
            self.inner_dim = self.out_ch * self.resize_factor
        elif inner_dim_calculation == "in":
            self.inner_dim = self.in_ch * self.resize_factor
        else:
            raise Exception(f"Wrong inner_dim_calculation: {inner_dim_calculation}")
            
        self.filter_per_group = filter_per_group
        self.se_type = se_type
        
        if se_type == "simple":
            se_constructor = SELayer
        elif se_type == "complex":
            se_constructor = SELayerComplex
        elif se_type == "none":
            se_constructor = lambda *args, **kwargs: nn.Identity()
        else:
            raise Exception(f"Wrong se_type: {se_type}")
            
        
        
        block = nn.Sequential(
                        nn.Conv1d(
                            in_channels=self.in_ch,
                            out_channels=self.inner_dim,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.inner_dim,
                            kernel_size=ks,
                            groups=self.inner_dim // self.filter_per_group,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.inner_dim),
                       activation(),
                       se_constructor(self.in_ch, 
                                      self.inner_dim,
                                      reduction=self.se_reduction), # self.in_ch is not good
                       nn.Conv1d(
                            in_channels=self.inner_dim,
                            out_channels=self.in_ch,
                            kernel_size=1,
                            padding='same',
                            bias=False
                       ),
                       nn.BatchNorm1d(self.in_ch),
                       activation(),
        )
        
      
        self.block = block
    
    def forward(self, x):
        return self.block(x)

class MappingBlock(nn.Module):
    def __init__(self, in_ch, out_ch, activation):
        super().__init__()
        self.block =  nn.Sequential(
                        nn.Conv1d(
                            in_channels=in_ch,
                            out_channels=out_ch,
                            kernel_size=1,
                            padding='same',
                       ),
                       activation()
        )
        
    def forward(self, x):
        return self.block(x)
    
class LegNet(nn.Module):
    """
    NoGINet neural network.

    Parameters
    ----------
    use_single_channel : bool
        If True, singleton channel is used.
    block_sizes : list, optional
        List containing block sizes. The default is [256, 256, 128, 128, 64, 64, 32, 32].
    ks : int, optional
        Kernel size of convolutional layers. The default is 5.
    resize_factor : int, optional
        Resize factor used in a high-dimensional middle layer of an EffNet-like block. The default is 4.
    activation : nn.Module, optional
        Activation function. The default is nn.SiLU.
    filter_per_group : int, optional
        Number of filters per group in a middle convolutiona layer of an EffNet-like block. The default is 2.
    se_reduction : int, optional
        Reduction number used in SELayer. The default is 4.
    final_ch : int, optional
        Number of channels in the final output convolutional channel. The default is 18.
    bn_momentum : float, optional
        BatchNorm momentum. The default is 0.1.

    """
    __constants__ = ('resize_factor')
    
    def __init__(self, 
                blocks: list[int]=[512, 256, 128, 128, 64, 64, 32, 32], 
                ks: int=7, 
                resize_factor: int=4, 
                activation: Type[nn.Module]=nn.SiLU,
                final_activation: Type[nn.Module]=nn.SiLU,
                filter_per_group: int=2,
                se_reduction: int=4,
                res_block_type: str="concat",
                se_type: str="complex",
                inner_dim_calculation: str="out"):        
        super().__init__()
        self.block_sizes = blocks
        self.resize_factor = resize_factor
        self.se_reduction = se_reduction
        self.filter_per_group = filter_per_group
        self.final_ch = 13 # number of bins in the competition
        self.inner_dim_calculation= inner_dim_calculation
        self.res_block_type = res_block_type
        self.in_channels = 4
        

        if res_block_type == "concat":
            residual = ResidualConcat
            local_multiplier = 2
        elif res_block_type == "add":
            residual = Residual
            local_multiplier = 1
        elif res_block_type == "none":
            residual = lambda x : x
            local_multiplier = 1
        else:
            raise NotImplementedError()    
        
        self.stem_block = LocalBlock(in_ch=self.in_channels,
                           out_ch=self.block_sizes[0],
                           ks=ks,
                           activation=activation)

        blocks = []
        for ind, (prev_sz, sz) in enumerate(zip(self.block_sizes[:-1], self.block_sizes[1:])):
            block = nn.Sequential(
                residual(EffBlock(in_ch=prev_sz, 
                         out_ch=sz,
                         ks=ks,
                         resize_factor=4,
                         activation=activation,
                         filter_per_group=self.filter_per_group,
                         se_type=se_type,
                         inner_dim_calculation=inner_dim_calculation)),
                LocalBlock(in_ch=local_multiplier * prev_sz,
                               out_ch=sz,
                               ks=ks,
                               activation=activation)
            )
            blocks.append(block)

        
        self.main = nn.Sequential(*blocks)

        self.mapper =  MappingBlock(in_ch=self.block_sizes[-1],
                                    out_ch=self.final_ch,
                                    activation=final_activation)
        
        
        self.register_buffer('bins', torch.arange(start=0, end=13, step=1, requires_grad=False))
    
    def forward(self, x):    
        x = x.permute(0,2,1)
        x = self.stem_block(x)
        x = self.main(x)
        x = self.mapper(x)
        x = F.adaptive_avg_pool1d(x, 1)
        x = x.squeeze(2)
        logprobs = F.log_softmax(x, dim=1) 
        x = F.softmax(x, dim=1)
        score = (x * self.bins).sum(dim=1)
        return logprobs, score
       