"""Neural models for the Burgers and Poisson comparisons."""

from __future__ import annotations

import os

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import numpy as np
import torch
import torch.nn as nn


def set_seed(seed: int = 1234) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)


class MLP(nn.Module):
    def __init__(self, in_dim: int = 2, out_dim: int = 1, width: int = 64, depth: int = 4):
        super().__init__()
        layers: list[nn.Module] = []
        last = in_dim
        for _ in range(depth):
            layers.append(nn.Linear(last, width))
            layers.append(nn.Tanh())
            last = width
        layers.append(nn.Linear(last, out_dim))
        self.net = nn.Sequential(*layers)
        self.apply(self._init)

    @staticmethod
    def _init(module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        return self.net(xt)


class QAPINN(nn.Module):
    """Hybrid quantum-assisted PINN.

    A small classical preprocessing layer maps (x,t) to qubit angles.  A
    parameterized quantum circuit produces Pauli-Z expectation values, then a
    classical head maps those quantum features to u(x,t).
    """

    def __init__(
        self,
        n_qubits: int = 3,
        q_layers: int = 2,
        hidden: int = 24,
        head_depth: int = 2,
    ):
        super().__init__()
        try:
            import pennylane as qml
        except Exception as exc:  # pragma: no cover - exercised only if env lacks PennyLane
            raise RuntimeError(
                "PennyLane could not be imported. Set NUMBA_CACHE_DIR to a writable "
                "directory or install pennylane."
            ) from exc

        dev = qml.device("default.qubit", wires=n_qubits)

        @qml.qnode(dev, interface="torch", diff_method="backprop")
        def circuit(inputs, weights):
            qml.AngleEmbedding(inputs, wires=range(n_qubits))
            qml.BasicEntanglerLayers(weights, wires=range(n_qubits))
            return [qml.expval(qml.PauliZ(i)) for i in range(n_qubits)]

        self.encoder = nn.Sequential(nn.Linear(2, n_qubits), nn.Tanh())
        self.quantum = qml.qnn.TorchLayer(circuit, {"weights": (q_layers, n_qubits)})
        head_layers: list[nn.Module] = []
        last = n_qubits
        for _ in range(head_depth):
            head_layers.append(nn.Linear(last, hidden))
            head_layers.append(nn.Tanh())
            last = hidden
        head_layers.append(nn.Linear(last, 1))
        self.head = nn.Sequential(*head_layers)
        self.head.apply(MLP._init)
        self.encoder.apply(MLP._init)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        angles = np.pi * self.encoder(xt)
        q_features = self.quantum(angles)
        if q_features.ndim == 1:
            q_features = q_features.unsqueeze(0)
        return self.head(q_features)


class XPINN(nn.Module):
    """Three-network XPINN with either time or space domain decomposition."""

    def __init__(
        self,
        split: str,
        cuts: tuple[float, float],
        width: int = 48,
        depth: int = 3,
    ):
        super().__init__()
        if split not in {"time", "space"}:
            raise ValueError("split must be 'time' or 'space'")
        self.split = split
        self.cuts = cuts
        self.subnets = nn.ModuleList([MLP(width=width, depth=depth) for _ in range(3)])

    def subnet_index(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        coord = t if self.split == "time" else x
        c1, c2 = self.cuts
        return torch.where(coord < c1, 0, torch.where(coord < c2, 1, 2)).long().squeeze(-1)

    def forward(self, xt: torch.Tensor) -> torch.Tensor:
        x = xt[:, :1]
        t = xt[:, 1:2]
        idx = self.subnet_index(x, t)
        out = torch.empty((xt.shape[0], 1), dtype=xt.dtype, device=xt.device)
        for k, net in enumerate(self.subnets):
            mask = idx == k
            if torch.any(mask):
                out[mask] = net(xt[mask])
        return out

    def eval_subnet(self, k: int, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        return self.subnets[k](torch.cat([x, t], dim=1))


def burgers_residual(model: nn.Module, x: torch.Tensor, t: torch.Tensor, nu: float) -> torch.Tensor:
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model(torch.cat([x, t], dim=1))
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    return u_t + u * u_x - nu * u_xx


def subnet_residual(model: XPINN, k: int, x: torch.Tensor, t: torch.Tensor, nu: float):
    x = x.clone().detach().requires_grad_(True)
    t = t.clone().detach().requires_grad_(True)
    u = model.eval_subnet(k, x, t)
    u_t = torch.autograd.grad(u, t, torch.ones_like(u), create_graph=True)[0]
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    f = u_t + u * u_x - nu * u_xx
    return f, u, u_x, u_t


def tt_svd_compress_vector(vec: np.ndarray, max_rank: int = 8) -> tuple[np.ndarray, int, float]:
    """Compress a length-2^n vector with tensor-train SVD and reconstruct it."""

    vec = np.asarray(vec, dtype=np.float64).ravel()
    n = int(np.log2(vec.size))
    if 2**n != vec.size:
        raise ValueError("TT-SVD helper expects vector length to be a power of two")
    tensor = vec.reshape([2] * n)
    ranks = [1]
    cores = []
    unfolding = tensor
    for site in range(n - 1):
        unfolding = unfolding.reshape(ranks[-1] * 2, -1)
        u, s, vh = np.linalg.svd(unfolding, full_matrices=False)
        r = min(max_rank, s.size)
        cores.append((u[:, :r]).reshape(ranks[-1], 2, r))
        unfolding = np.diag(s[:r]) @ vh[:r, :]
        ranks.append(r)
    cores.append(unfolding.reshape(ranks[-1], 2, 1))

    rec = cores[0]
    for core in cores[1:]:
        rec = np.tensordot(rec, core, axes=([-1], [0]))
    rec = rec.reshape(vec.shape)
    rel_err = float(np.linalg.norm(rec - vec) / max(np.linalg.norm(vec), 1e-12))
    return rec, max(ranks), rel_err
