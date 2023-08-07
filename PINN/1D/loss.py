import torch
import torch.nn.functional as F
from torch.autograd import grad

tensor = torch.Tensor



def bc_derivative_mse(
    x1: tensor, t1: tensor, field1: tensor, x2: tensor, t2: tensor, field2: tensor
) -> tensor:
    """
    MSE loss of derivatives at boundary:
    u_x(left) = u_x(right)
    u_t(left) = u_t(right)

    Args:
    x, t, field: position, time, field at boundary
    x, y, t: [num_data, 1]
    field: [num_data, 1]
    """
    u1_x = grad(field1, x1, torch.ones_like(field1), create_graph=True)[0]
    u1_t = grad(field1, t1, torch.ones_like(field1), create_graph=True)[0]

    u2_x = grad(field2, x2, torch.ones_like(field2), create_graph=True)[0]
    u2_t = grad(field2, t2, torch.ones_like(field2), create_graph=True)[0]

    bc_derivative1 = torch.cat([u1_x, u1_t], dim=-1)
    bc_derivative2 = torch.cat([u2_x, u2_t], dim=-1)

    return F.mse_loss(bc_derivative1, bc_derivative2)



def pde_mse(x: tensor, t: tensor, field: tensor, nu: tensor) -> tensor:
    """
    MSE loss of burgers equation
    u_t + u * u_x - nu * u_xx = 0

    Args
    x, t, field: position, time, field at boundary
    x, t: [num_data, 1]
    field: [num_data, 1]
    nu: [1, ]
    """
    u_x = grad(field, x, torch.ones_like(field), create_graph=True)[0]
    u_t = grad(field, t, torch.ones_like(field), create_graph=True)[0]

    u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]

    return (u_t + field * u_x - nu * u_xx).square().mean()
