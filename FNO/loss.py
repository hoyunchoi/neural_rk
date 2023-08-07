import torch


# loss function with rel/abs Lp loss
class LpLoss:
    def __init__(
        self, d: int = 2, p: int = 2, size_average: bool = True, reduction: bool = True
    ):
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.reduction = reduction
        self.size_average = size_average

    def abs(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_batch = x.shape[0]

        # Assume uniform mesh
        h = 1.0 / (x.shape[1] - 1.0)
        all_norms = (h ** (self.d / self.p)) * torch.linalg.vector_norm(
            x.view(num_batch, -1) - y.view(num_batch, -1), ord=self.p, dim=1
        )
        if not self.reduction:
            return all_norms

        if self.size_average:
            return torch.mean(all_norms)
        else:
            return torch.sum(all_norms)

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_batch = x.shape[0]

        diff_norms = torch.linalg.vector_norm(
            x.reshape(num_batch, -1) - y.reshape(num_batch, -1), ord=self.p, dim=1
        )
        y_norms = torch.linalg.vector_norm(y.reshape(num_batch, -1), ord=self.p, dim=1)

        if not self.reduction:
            return diff_norms / y_norms

        if self.size_average:
            return torch.mean(diff_norms / y_norms)
        else:
            return torch.sum(diff_norms / y_norms)

    def __call__(self, x, y):
        return self.rel(x, y)


# Sobolev norm (HS norm)
# where we also compare the numerical derivatives between the output and target
class HsLoss:
    def __init__(
        self,
        d: int = 2,
        p: int = 2,
        k: int = 1,
        a: list[int] | None = None,
        group: bool = False,
        size_average: bool = True,
        reduction: bool = True,
    ) -> None:
        # Dimension and Lp-norm type are postive
        assert d > 0 and p > 0

        self.d = d
        self.p = p
        self.k = k
        self.balanced = group
        self.reduction = reduction
        self.size_average = size_average

        if a is None:
            a = [1] * k
        self.a = a

    def rel(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        num_batch = x.shape[0]
        diff_norms = torch.linalg.norm(
            x.reshape(num_batch, -1) - y.reshape(num_batch, -1), self.p, 1
        )
        y_norms = torch.linalg.norm(y.reshape(num_batch, -1), self.p, 1)
        if not self.reduction:
            return diff_norms / y_norms

        if self.size_average:
            return torch.mean(diff_norms / y_norms)
        else:
            return torch.sum(diff_norms / y_norms)

    def __call__(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        nx = x.shape[1]
        ny = x.shape[2]
        k = self.k
        a = self.a
        x = x.view(x.shape[0], nx, ny, -1)
        y = y.view(y.shape[0], nx, ny, -1)

        k_x = (
            torch.cat(
                (torch.arange(0, nx // 2), torch.arange(-nx // 2, 0)),
                dim=0,
            )
            .reshape(nx, 1)
            .repeat(1, ny)
        )
        k_y = (
            torch.cat(
                (torch.arange(0, ny // 2), torch.arange(-ny // 2, 0)),
                dim=0,
            )
            .reshape(1, ny)
            .repeat(nx, 1)
        )
        k_x = torch.abs(k_x).reshape(1, nx, ny, 1).to(x.device)
        k_y = torch.abs(k_y).reshape(1, nx, ny, 1).to(x.device)

        x = torch.fft.fftn(x, dim=[1, 2])
        y = torch.fft.fftn(y, dim=[1, 2])

        if self.balanced:
            loss = self.rel(x, y)
            if k >= 1:
                weight = a[0] * torch.sqrt(k_x**2 + k_y**2)
                loss += self.rel(x * weight, y * weight)
            if k >= 2:
                weight = a[1] * torch.sqrt(
                    k_x**4 + 2 * k_x**2 * k_y**2 + k_y**4
                )
                loss += self.rel(x * weight, y * weight)
            loss = loss / (k + 1)

        else:
            weight = torch.tensor(1.0, device=x.device)
            if k >= 1:
                weight += a[0] ** 2 * (k_x**2 + k_y**2)
            if k >= 2:
                weight += a[1] ** 2 * (k_x**4 + 2 * k_x**2 * k_y**2 + k_y**4)
            weight = torch.sqrt(weight)
            loss = self.rel(x * weight, y * weight)

        return loss


# # A simple feedforward neural network
# class DenseNet(nn.Module):
#     def __init__(self, layers: list[int], nonlinearity, out_nonlinearity=None, normalize: bool=False) -> None:
#         super().__init__()

#         self.n_layers = len(layers) - 1

#         assert self.n_layers >= 1

#         self.layers = nn.ModuleList()

#         for j in range(self.n_layers):
#             self.layers.append(nn.Linear(layers[j], layers[j + 1]))

#             if j != self.n_layers - 1:
#                 if normalize:
#                     self.layers.append(nn.BatchNorm1d(layers[j + 1]))

#                 self.layers.append(nonlinearity())

#         if out_nonlinearity is not None:
#             self.layers.append(out_nonlinearity())

#     def forward(self, x):
#         for _, l in enumerate(self.layers):
#             x = l(x)

#         return x


# # print the number of parameters
# def count_params(model):
#     c = 0
#     for p in list(model.parameters()):
#         c += reduce(operator.mul, list(p.size() + (2,) if p.is_complex() else p.size()))
#     return c
