# utils/fisher.py
import torch
import torch.nn.functional as F

def sphere_map(p, eps=1e-8):
    # p: (..., K), simplex probs
    p = torch.clamp(p, min=eps)
    s = torch.sqrt(p)
    # 单位球面，数值上稳，且对 one-hot 映射仍是基向量
    return F.normalize(s, p=2, dim=-1)

def inv_sphere_map(s, eps=1e-8):
    # s: (..., K), on sphere
    s = torch.clamp(s, min=0)
    p = s ** 2
    return p / p.sum(-1, keepdim=True).clamp_min(eps)

def proj_tangent(x, v):
    # 投影到切空间 T_x S：v - <v,x> x
    return v - (v * x).sum(-1, keepdim=True) * x

def slerp(x0, x1, t):
    # 球面线性插值（测地线），t: (B,) or (B,L)
    dot = (x0 * x1).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    theta = torch.acos(dot)
    sin_theta = torch.sin(theta).clamp_min(1e-6)

    if t.ndim == 1:
        t = t.view(-1, 1, 1)
    else:
        t = t.unsqueeze(-1)

    a = torch.sin((1 - t) * theta) / sin_theta
    b = torch.sin(t * theta) / sin_theta
    x = a * x0 + b * x1
    return F.normalize(x, p=2, dim=-1)

def logmap_sphere(x, y):
    # log_x(y) = (α / sin α) * (y - cos α * x), α = arccos(<x,y>)
    dot = (x * y).sum(-1, keepdim=True).clamp(-1 + 1e-6, 1 - 1e-6)
    alpha = torch.acos(dot)
    sin_alpha = torch.sin(alpha).clamp_min(1e-6)
    return (alpha / sin_alpha) * (y - dot * x)