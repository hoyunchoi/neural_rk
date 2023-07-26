import torch
import torch.nn.functional as F
from torch.autograd import grad

tensor = torch.Tensor


def bc_derivative_mse(
    x1: tensor,
    y1: tensor,
    t1: tensor,
    field1: tensor,
    x2: tensor,
    y2: tensor,
    t2: tensor,
    field2: tensor,
) -> tensor:
    """
    MSE loss of derivatives at boundary:
    u_x(left) = u_x(right), u_x(bottom) = u_x(top)
    u_y(left) = u_y(right), u_y(bottom) = u_y(top)
    u_t(left) = u_t(right), u_t(bottom) = u_t(top)
    v_x(left) = v_x(right), v_x(bottom) = v_x(top)
    v_y(left) = v_y(right), v_y(bottom) = v_y(top)
    v_t(left) = v_t(right), v_t(bottom) = v_t(top)

    Args:
    x, y, t, field: positions, time, field at boundary
    x, y, t: [num_data, 1]
    field: [num_data, 2]
    """
    u1, v1 = field1.T
    u1, v1 = u1.unsqueeze(-1), v1.unsqueeze(-1)
    u2, v2 = field2.T
    u2, v2 = u2.unsqueeze(-1), v2.unsqueeze(-1)

    u1_x = grad(u1, x1, torch.ones_like(u1), create_graph=True)[0]
    u1_y = grad(u1, y1, torch.ones_like(u1), create_graph=True)[0]
    u1_t = grad(u1, t1, torch.ones_like(u1), create_graph=True)[0]
    v1_x = grad(v1, x1, torch.ones_like(v1), create_graph=True)[0]
    v1_y = grad(v1, y1, torch.ones_like(v1), create_graph=True)[0]
    v1_t = grad(v1, t1, torch.ones_like(v1), create_graph=True)[0]

    u2_x = grad(u2, x2, torch.ones_like(u2), create_graph=True)[0]
    u2_y = grad(u2, y2, torch.ones_like(u2), create_graph=True)[0]
    u2_t = grad(u2, t2, torch.ones_like(u2), create_graph=True)[0]
    v2_x = grad(v2, x2, torch.ones_like(v2), create_graph=True)[0]
    v2_y = grad(v2, y2, torch.ones_like(v2), create_graph=True)[0]
    v2_t = grad(v2, t2, torch.ones_like(v2), create_graph=True)[0]

    bc_derivative1 = torch.cat([u1_x, u1_y, u1_t, v1_x, v1_y, v1_t], dim=-1)
    bc_derivative2 = torch.cat([u2_x, u2_y, u2_t, v2_x, v2_y, v2_t], dim=-1)

    return F.mse_loss(bc_derivative1, bc_derivative2)


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
