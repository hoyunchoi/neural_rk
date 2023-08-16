import torch
import torch.nn.functional as F
from torch.autograd import grad

tensor = torch.Tensor



def pde_mse(x: tensor, y: tensor, t: tensor, field: tensor, nu: tensor) -> tensor:
    """
    MSE loss of burgers equation
    u_t + u * u_x + v * u_y - nu * (u_xx + u_yy) = 0
    v_t + u * v_x + v * v_y - nu * (v_xx + v_yy) = 0

    Args
    x, y, t, field: positions, time, field at boundary
    x, y, t: [num_data, 1]
    field: [num_data, 2]
    nu: [2, ]
    """
    u, v = field.T
    u, v = u.unsqueeze(-1), v.unsqueeze(-1)

    u_x = grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_t = grad(u, t, torch.ones_like(u), create_graph=True)[0]
    v_x = grad(v, x, torch.ones_like(v), create_graph=True)[0]
    v_y = grad(v, y, torch.ones_like(v), create_graph=True)[0]
    v_t = grad(v, t, torch.ones_like(v), create_graph=True)[0]

    u_xx = grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    v_xx = grad(v_x, x, torch.ones_like(v_x), create_graph=True)[0]
    v_yy = grad(v_y, y, torch.ones_like(v_y), create_graph=True)[0]

    loss_u = u_t + u * u_x + v * u_y - nu[0] * (u_xx + u_yy)
    loss_v = v_t + u * v_x + v * v_y - nu[1] * (v_xx + v_yy)

    return torch.cat((loss_u, loss_v), dim=-1).square().mean()


def pde_mse_1d(x: tensor, t: tensor, field: tensor, nu: tensor) -> tensor:
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
