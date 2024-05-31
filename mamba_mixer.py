import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def selective_scan_ref(u: torch.Tensor, delta: torch.Tensor, A: torch.Tensor, B: torch.Tensor, C: torch.Tensor, 
                       D: torch.Tensor = None, z: torch.Tensor = None, delta_bias: torch.Tensor = None, 
                       delta_softplus: bool = False, return_last_state: bool = False):
    """
    Perform selective scanning over the input sequences.
    """
    u = u.float()
    delta = delta.float()
    
    if delta_bias is not None:
        delta += delta_bias[..., None].float()
        
    if delta_softplus:
        delta = F.softplus(delta)
        
    batch, dim, seq_len = u.shape[0], A.shape[0], u.shape[2]
    dstate = A.shape[1]
    
    x = torch.zeros(batch, dim, dstate, device=u.device)
    ys = []

    deltaA = torch.einsum("bdl,dn->bdln", delta, A).exp()

    if B.dim() == 2:
        deltaB_u = torch.einsum("bdl,dn,bdl->bdln", delta, B, u)
    elif B.dim() == 3:
        deltaB_u = torch.einsum("bdl,bnl,bdl->bdln", delta, B, u)
    else:
        B = B.repeat(1, dim // B.shape[1], 1, 1)
        deltaB_u = torch.einsum("bdl,bdnl,bdl->bdln", delta, B, u)

    if C.dim() == 4:
        C = C.repeat(1, dim // C.shape[1], 1, 1)

    for i in range(seq_len):
        x = deltaA[:, :, i] * x + deltaB_u[:, :, i]
        if C.dim() == 2:
            y = torch.einsum("bdn,dn->bd", x, C)
        elif C.dim() == 3:
            y = torch.einsum("bdn,bn->bd", x, C[:, :, i])
        else:
            y = torch.einsum("bdn,bdn->bd", x, C[:, :, :, i])
            
        ys.append(y)

    y = torch.stack(ys, dim=2)  # (batch, dim, seq_len)
    out = y if D is None else y + u * D.view(-1, 1)
    
    if z is not None:
        out *= z.silu()
        
    return out if not return_last_state else (out, x)

class MambaMixer(nn.Module):
    def __init__(self, dim: int, d_state: int = 16, d_conv: int = 4, expand: int = 2, dt_rank: str = "auto", 
                 dt_min: float = 0.001, dt_max: float = 0.1, dt_init: str = "random", dt_scale: float = 1.0, 
                 dt_init_floor: float = 1e-4, conv_bias: bool = True, bias: bool = False, layer_idx: int = None):
        super(MambaMixer, self).__init__()
        
        self.dim = dim
        self.d_state = d_state
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = self.expand * self.dim
        self.dt_rank = math.ceil(self.dim / 16) if dt_rank == "auto" else dt_rank
        self.layer_idx = layer_idx
        
        self.in_proj = nn.Linear(self.dim, self.d_inner * 2, bias=bias)
        self.conv1d = nn.Conv1d(self.d_inner, self.d_inner, kernel_size=d_conv, groups=self.d_inner, 
                                padding=d_conv - 1, bias=conv_bias)
        
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + self.d_state * 2, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)
        
        dt_init_std = self.dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            self.dt_proj.weight.data.fill_(dt_init_std)
        elif dt_init == "random":
            self.dt_proj.weight.data.uniform_(-dt_init_std, dt_init_std)
        else:
            raise NotImplementedError(f"Unsupported dt_init value: {dt_init}")
        
        dt = torch.logspace(math.log10(dt_min), math.log10(dt_max), self.d_inner).clamp_min(dt_init_floor)
        inv_dt = dt + torch.log1p(-torch.exp(-dt))
        self.dt_proj.bias.data = inv_dt
        
        self.A_log = torch.arange(1, self.d_state + 1).repeat(self.d_inner, 1).log()
        self.D = nn.Parameter(torch.ones(self.d_inner))
        self.out_proj = nn.Linear(self.d_inner, self.dim, bias=bias)
    
    def forward(self, hidden_states: torch.Tensor, inference_params: dict = None):
        batch, seqlen, dim = hidden_states.shape
        conv_state, ssm_state = None, None
        
        if inference_params is not None:
            conv_state, ssm_state = self._get_states_from_cache(inference_params, batch)
            if inference_params.get("seqlen_offset", 0) > 0:
                out, _, _ = self.step(hidden_states[:, -1:, :], conv_state, ssm_state)
                return out
        
        xz = self.in_proj(hidden_states)
        x, z = xz.chunk(2, dim=-1)
        
        if conv_state is not None:
            conv_state = x[:, -self.d_conv:, :]  # Update state (B, D, W)
            x = self.conv1d(x.transpose(1, 2))[..., :seqlen].transpose(1, 2).silu()
        
        x_db = self.x_proj(x)
        dt, B, C = x_db.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)
        
        B = B.view(batch * seqlen, self.d_inner, self.d_state)
        C = C.view(batch * seqlen, self.d_inner, self.d_state)
        
        A = -self.A_log.exp()
        
        y = selective_scan_ref(x.view(batch * seqlen, self.d_inner, 1), 
                               dt.view(batch * seqlen, self.d_inner, 1), 
                               A, B, C, self.D, 
                               z=z.view(batch * seqlen, self.d_inner, 1), 
                               delta_bias=self.dt_proj.bias, 
                               delta_softplus=True, 
                               return_last_state=ssm_state is not None)
        
        if ssm_state is not None:
            y, last_state = y
            ssm_state = last_state
        
        y = y.view(batch, seqlen, self.d_inner)
        out = self.out_proj(y)
        
        return out
    
    def step(self, hidden_states: torch.Tensor, conv_state: torch.Tensor, ssm_state: torch.Tensor):
        assert hidden_states.size(1) == 1, "Only support decoding with 1 token at a time"
        
        xz = self.in_proj(hidden_states.squeeze(1))  # (B, 2D)
        x, z = xz.chunk(2, dim=-1)  # (B, D)
        
        conv_state = torch.cat([conv_state[:, 1:, :], x.unsqueeze(1)], dim=1)
        x = F.conv1d(conv_state.transpose(1, 2), self.conv1d.weight, self.conv1d.bias).transpose(1, 2).silu()
        
        x_db = self.x_proj(x)  # (B, dt_rank + 2*d_state)
        dt, B, C = x_db.split([self.dt_rank, self.d_state, self.d_state], dim=-1)
        dt = self.dt_proj(dt)  # Don't add dt_bias here
        
        A = -self.A_log.exp()
        dt = (dt + self.dt_proj.bias.unsqueeze(0)).softplus()
        
        dA = torch.einsum("db,dn->bdn", dt, A).exp()
        dB = torch.einsum("db,bn->bdn", dt, B)
        ssm_state = ssm_state * dA + x.unsqueeze(-1) * dB
        
        y = torch.einsum("bdn,bn->bd", ssm_state, C)
        y += self.D * x
        y *= z.silu()  # (B, D)
        
        out = self.out_proj(y)
        return out.unsqueeze(1), conv_state, ssm_state
    
    def _get_states_from_cache(self, inference_params: dict, batch_size: int, initialize_states: bool = False):
        assert self.layer_idx is not None, "Layer index must be set"
        
        if self.layer_idx not in inference_params:
            conv_state = torch.zeros(batch_size, self.d_conv, self.d_inner, device=inference_params['device'])
            ssm_state = torch.zeros(batch_size, self.d_inner, self.d_state, device=inference_params['device'])
            inference_params[self.layer_idx] = (conv_state, ssm_state)
        else:
            conv_state, ssm_state = inference_params[self.layer_idx]
        
        return conv_state, ssm_state
