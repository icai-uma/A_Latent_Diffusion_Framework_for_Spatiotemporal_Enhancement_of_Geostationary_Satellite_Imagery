import torch


def slerp_im( img1, img2,t):
    flat1, flat2 = img1.flatten(), img2.flatten()
    norm1 = flat1 / flat1.norm()
    norm2 = flat2 / flat2.norm()
    dot = (norm1 * norm2).sum()
    omega = torch.acos(torch.clamp(dot, -1, 1))
    if omega.abs() < 1e-5:
        flat_interp = (1 - t) * flat1 + t * flat2
    else:
        so = torch.sin(omega)
        flat_interp = (
            torch.sin((1.0 - t) * omega) / so * flat1 +
            torch.sin(t * omega) / so * flat2
        )
    return flat_interp.view_as(img1)
    
    
def slerp(val_0, val_1, t):

    dot = torch.sum(val_0 * val_1, dim=-1, keepdim=True) / (torch.norm(val_0) * torch.norm(val_1))
    dot = torch.clamp(dot, -1.0, 1.0)  
    
    theta = torch.acos(dot)  
    
    sin_theta = torch.sin(theta)
    
    factor_0 = torch.sin((1.0 - t) * theta) / sin_theta
    factor_1 = torch.sin(t * theta) / sin_theta
    
    return factor_0 * val_0 + factor_1 * val_1