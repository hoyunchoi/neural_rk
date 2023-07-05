import numpy as np
import numpy.typing as npt
import torch


class SigmoidStandardScaler:
    def __init__(self) -> None:
        """
        Initialize mean, standard deviation stored in scaler
        Both of them don't have any gradient
        """
        self.avg: torch.Tensor
        self.std: torch.Tensor

    def fit(self, data: npt.NDArray[np.float32] | torch.Tensor) -> None:
        data = torch.as_tensor(data)
        self.std, self.avg = torch.std_mean(torch.sigmoid(data), dim=0, unbiased=False)

    def transform(self, data: npt.NDArray[np.float32] | torch.Tensor) -> torch.Tensor:
        data = torch.as_tensor(data)
        self.avg = self.avg.to(data.device, data.dtype)
        self.std = self.std.to(data.device, data.dtype)
        return (torch.sigmoid(data) - self.avg) / self.std

    def fit_transform(self, data: npt.NDArray[np.float32] | torch.Tensor) -> torch.Tensor:
        self.fit(data)
        return self.transform(data)

    def inverse_transform(self, data: npt.NDArray[np.float32] | torch.Tensor) -> torch.Tensor:
        data = torch.as_tensor(data)
        self.avg = self.avg.to(data.device, data.dtype)
        self.std = self.std.to(data.device, data.dtype)
        return torch.logit((data * self.std) + self.avg)
