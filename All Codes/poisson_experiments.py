"""Poisson-equation comparison matching the Burgers experiment structure.

Problem:
    u_xx + u_yy = -2*pi^2*sin(pi*x)*sin(pi*y),  (x,y) in [0,1]^2
    u = 0 on the boundary

Exact solution:
    u(x,y) = sin(pi*x)*sin(pi*y)
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", "/tmp/mpl")
os.environ.setdefault("NUMBA_CACHE_DIR", "/tmp/numba")

import matplotlib.pyplot as plt
import numpy as np
import torch

from burgers_models import MLP, QAPINN, set_seed
from burgers_reference import grid_metrics


def exact_u(x, y):
    return np.sin(np.pi * x) * np.sin(np.pi * y)


def forcing_torch(x, y):
    return -2.0 * torch.pi**2 * torch.sin(torch.pi * x) * torch.sin(torch.pi * y)


def tensor(a, device):
    return torch.as_tensor(a, dtype=torch.float32, device=device)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def poisson_residual(model, x, y):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    u = model(torch.cat([x, y], dim=1))
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy - forcing_torch(x, y)


def sample_pinn(n_f, n_b, device):
    x_f = torch.rand(n_f, 1, device=device)
    y_f = torch.rand(n_f, 1, device=device)
    s = torch.rand(n_b, 1, device=device)
    side = torch.randint(0, 4, (n_b, 1), device=device)
    x_b = torch.where(side == 0, torch.zeros_like(s), torch.where(side == 1, torch.ones_like(s), s))
    y_b = torch.where(side == 2, torch.zeros_like(s), torch.where(side == 3, torch.ones_like(s), s))
    u_b = torch.zeros_like(x_b)
    return x_f, y_f, x_b, y_b, u_b


def train_pinn_like(model, label, args, device, epochs=None):
    epochs = int(args.epochs if epochs is None else epochs)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = {"loss_total": [], "loss_pde": [], "loss_bc": []}
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        x_f, y_f, x_b, y_b, u_b = sample_pinn(args.n_f, args.n_b, device)
        r = poisson_residual(model, x_f, y_f)
        loss_pde = torch.mean(r**2)
        loss_bc = torch.mean((model(torch.cat([x_b, y_b], dim=1)) - u_b) ** 2)
        loss = loss_pde + args.bc_weight * loss_bc
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        hist["loss_total"].append(float(loss.detach().cpu()))
        hist["loss_pde"].append(float(loss_pde.detach().cpu()))
        hist["loss_bc"].append(float(loss_bc.detach().cpu()))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == epochs:
            print(f"{label:18s} epoch {epoch:5d}/{epochs} loss={hist['loss_total'][-1]:.3e}")
    hist["wall_time_sec"] = time.perf_counter() - start
    hist["epochs"] = epochs
    return hist


class PoissonXPINN(torch.nn.Module):
    def __init__(self, split="x", cuts=(1.0 / 3.0, 2.0 / 3.0), width=40, depth=3):
        super().__init__()
        self.split = split
        self.cuts = cuts
        self.subnets = torch.nn.ModuleList([MLP(width=width, depth=depth) for _ in range(3)])

    def subnet(self, k, x, y):
        return self.subnets[k](torch.cat([x, y], dim=1))

    def forward(self, xy):
        x, y = xy[:, :1], xy[:, 1:2]
        coord = x if self.split == "x" else y
        c1, c2 = self.cuts
        idx = torch.where(coord < c1, 0, torch.where(coord < c2, 1, 2)).long().squeeze(1)
        out = torch.empty((xy.shape[0], 1), dtype=xy.dtype, device=xy.device)
        for k, net in enumerate(self.subnets):
            mask = idx == k
            if torch.any(mask):
                out[mask] = net(xy[mask])
        return out


def subnet_residual(model, k, x, y):
    x = x.clone().detach().requires_grad_(True)
    y = y.clone().detach().requires_grad_(True)
    u = model.subnet(k, x, y)
    u_x = torch.autograd.grad(u, x, torch.ones_like(u), create_graph=True)[0]
    u_y = torch.autograd.grad(u, y, torch.ones_like(u), create_graph=True)[0]
    u_xx = torch.autograd.grad(u_x, x, torch.ones_like(u_x), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones_like(u_y), create_graph=True)[0]
    return u_xx + u_yy - forcing_torch(x, y), u, u_x, u_y


def subdomain_ranges(split, cuts):
    c1, c2 = cuts
    if split == "x":
        return [((0.0, c1), (0.0, 1.0)), ((c1, c2), (0.0, 1.0)), ((c2, 1.0), (0.0, 1.0))]
    return [((0.0, 1.0), (0.0, c1)), ((0.0, 1.0), (c1, c2)), ((0.0, 1.0), (c2, 1.0))]


def sample_rect(n, xr, yr, device):
    x = xr[0] + (xr[1] - xr[0]) * torch.rand(n, 1, device=device)
    y = yr[0] + (yr[1] - yr[0]) * torch.rand(n, 1, device=device)
    return x, y


def train_xpinn(split, args, device):
    model = PoissonXPINN(split=split, width=args.xpinn_width, depth=args.xpinn_depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    ranges = subdomain_ranges(split, model.cuts)
    hist = {"loss_total": [], "loss_pde": [], "loss_bc": [], "loss_interface": []}
    start = time.perf_counter()
    for epoch in range(1, args.xpinn_epochs + 1):
        loss_pde = torch.zeros((), device=device)
        loss_bc = torch.zeros((), device=device)
        loss_interface = torch.zeros((), device=device)
        for k, (xr, yr) in enumerate(ranges):
            x_f, y_f = sample_rect(args.n_f // 3, xr, yr, device)
            r, _, _, _ = subnet_residual(model, k, x_f, y_f)
            loss_pde = loss_pde + torch.mean(r**2)

            n_edge = max(16, args.n_b // 8)
            # Global boundary edges belonging to the subdomain.
            if split == "x" and k == 0:
                yb = torch.rand(n_edge, 1, device=device); xb = torch.zeros_like(yb)
                loss_bc = loss_bc + torch.mean(model.subnet(k, xb, yb) ** 2)
            if split == "x" and k == 2:
                yb = torch.rand(n_edge, 1, device=device); xb = torch.ones_like(yb)
                loss_bc = loss_bc + torch.mean(model.subnet(k, xb, yb) ** 2)
            if split == "y" and k == 0:
                xb = torch.rand(n_edge, 1, device=device); yb = torch.zeros_like(xb)
                loss_bc = loss_bc + torch.mean(model.subnet(k, xb, yb) ** 2)
            if split == "y" and k == 2:
                xb = torch.rand(n_edge, 1, device=device); yb = torch.ones_like(xb)
                loss_bc = loss_bc + torch.mean(model.subnet(k, xb, yb) ** 2)

            # Horizontal/vertical outer boundaries that every slab touches.
            if split == "x":
                xb, _ = sample_rect(n_edge, xr, (0.0, 1.0), device)
                loss_bc = loss_bc + torch.mean(model.subnet(k, xb, torch.zeros_like(xb)) ** 2)
                loss_bc = loss_bc + torch.mean(model.subnet(k, xb, torch.ones_like(xb)) ** 2)
            else:
                _, yb = sample_rect(n_edge, (0.0, 1.0), yr, device)
                loss_bc = loss_bc + torch.mean(model.subnet(k, torch.zeros_like(yb), yb) ** 2)
                loss_bc = loss_bc + torch.mean(model.subnet(k, torch.ones_like(yb), yb) ** 2)

        for j, cut in enumerate(model.cuts):
            s = torch.rand(args.n_interface, 1, device=device)
            if split == "x":
                x_i = torch.full_like(s, cut); y_i = s
            else:
                x_i = s; y_i = torch.full_like(s, cut)
            r_l, u_l, ux_l, uy_l = subnet_residual(model, j, x_i, y_i)
            r_r, u_r, ux_r, uy_r = subnet_residual(model, j + 1, x_i, y_i)
            normal_jump = ux_l - ux_r if split == "x" else uy_l - uy_r
            loss_interface = loss_interface + torch.mean((u_l - u_r) ** 2)
            loss_interface = loss_interface + 0.1 * torch.mean(normal_jump**2)
            loss_interface = loss_interface + 0.01 * torch.mean((r_l - r_r) ** 2)

        loss = loss_pde + args.bc_weight * loss_bc + args.interface_weight * loss_interface
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        hist["loss_total"].append(float(loss.detach().cpu()))
        hist["loss_pde"].append(float(loss_pde.detach().cpu()))
        hist["loss_bc"].append(float(loss_bc.detach().cpu()))
        hist["loss_interface"].append(float(loss_interface.detach().cpu()))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.xpinn_epochs:
            print(f"XPINN-{split:5s}       epoch {epoch:5d}/{args.xpinn_epochs} loss={hist['loss_total'][-1]:.3e}")
    hist["wall_time_sec"] = time.perf_counter() - start
    hist["epochs"] = int(args.xpinn_epochs)
    return model, hist


@torch.no_grad()
def predict_grid(model, x, y, device):
    xx, yy = np.meshgrid(x, y, indexing="ij")
    xy = np.stack([xx.ravel(), yy.ravel()], axis=1).astype(np.float32)
    out = []
    model.eval()
    for i in range(0, len(xy), 8192):
        out.append(model(tensor(xy[i : i + 8192], device)).cpu().numpy())
    return np.concatenate(out).reshape(len(x), len(y))


def make_figures(outdir, x, y, u_ref, predictions, histories, metrics):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    yy, xx = np.meshgrid(y, x)
    for name, u_pred in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
        im0 = axes[0].pcolormesh(xx, yy, u_ref, shading="auto", cmap="viridis")
        axes[0].set_title("Exact")
        im1 = axes[1].pcolormesh(xx, yy, u_pred, shading="auto", cmap="viridis")
        axes[1].set_title(name)
        im2 = axes[2].pcolormesh(xx, yy, np.abs(u_pred - u_ref), shading="auto", cmap="magma")
        axes[2].set_title("|error|")
        for ax in axes:
            ax.set_xlabel("x"); ax.set_ylabel("y")
        fig.colorbar(im0, ax=axes[:2], shrink=0.85)
        fig.colorbar(im2, ax=axes[2], shrink=0.85)
        fig.savefig(figdir / f"{name.lower().replace('-', '_')}_field_error.png", dpi=220)
        plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    mid = len(y) // 2
    ax.plot(x, u_ref[:, mid], "k-", lw=2.5, label="Exact y=0.5")
    for name, u_pred in predictions.items():
        ax.plot(x, u_pred[:, mid], lw=1.5, label=name)
    ax.set_xlabel("x"); ax.set_ylabel("u(x,0.5)")
    ax.set_title("Poisson centerline comparison")
    ax.grid(alpha=0.25); ax.legend(fontsize=8, ncol=2)
    fig.savefig(figdir / "centerline_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    for name, hist in histories.items():
        ax.semilogy(hist["loss_total"], label=name)
    ax.set_xlabel("Epoch"); ax.set_ylabel("Total loss")
    ax.set_title("Poisson training losses")
    ax.grid(alpha=0.25); ax.legend()
    fig.savefig(figdir / "loss_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    labels = list(metrics)
    ax.bar(labels, [metrics[k]["rel_l2"] for k in labels])
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Poisson relative L2 comparison")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(figdir / "relative_l2_bar.png", dpi=220)
    plt.close(fig)

    metric_names = ["mse", "mae", "linf", "rel_l2"]
    labels = list(metrics)
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, metric in zip(axes.ravel(), metric_names):
        ax.bar(labels, [metrics[k][metric] for k in labels])
        ax.set_title(metric.upper())
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(figdir / "all_error_metrics_bar.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(14, 4), constrained_layout=True)
    for ax, key, title, ylabel in [
        (axes[0], "wall_time_sec", "Training time", "seconds"),
        (axes[1], "params", "Trainable parameters", "parameters"),
        (axes[2], "epochs", "Epoch budget", "epochs"),
    ]:
        ax.bar(labels, [metrics[k][key] for k in labels])
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.tick_params(axis="x", rotation=25)
        ax.grid(axis="y", alpha=0.25)
    fig.savefig(figdir / "cost_comparison.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()
    im = axes[0].pcolormesh(xx, yy, u_ref, shading="auto", cmap="viridis")
    axes[0].set_title("Exact")
    for ax, (name, pred) in zip(axes[1:], predictions.items()):
        ax.pcolormesh(xx, yy, pred, shading="auto", cmap="viridis")
        ax.set_title(name)
    for ax in axes:
        ax.set_xlabel("x")
        ax.set_ylabel("y")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    fig.savefig(figdir / "prediction_montage.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.4, 4.4), constrained_layout=True)
    for name, u_pred in predictions.items():
        ax.plot(x, np.abs(u_pred[:, mid] - u_ref[:, mid]), lw=1.6, label=name)
    ax.set_xlabel("x")
    ax.set_ylabel("|error| at y=0.5")
    ax.set_title("Poisson centerline absolute error")
    ax.grid(alpha=0.25)
    ax.legend(fontsize=8, ncol=2)
    fig.savefig(figdir / "centerline_error.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(12, 2.2), constrained_layout=True)
    rows = [
        [
            name,
            f"{metrics[name]['rel_l2']:.4f}",
            f"{metrics[name]['mse']:.3e}",
            f"{metrics[name]['wall_time_sec']:.1f}",
            f"{metrics[name]['epochs']}",
            f"{metrics[name]['params']:,}",
        ]
        for name in labels
    ]
    ax.axis("off")
    table = ax.table(
        cellText=rows,
        colLabels=["Model", "Rel L2", "MSE", "Time (s)", "Epochs", "Params"],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1, 1.4)
    fig.savefig(figdir / "summary_table.png", dpi=220)
    plt.close(fig)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--outdir", default="poisson_results")
    p.add_argument("--seed", type=int, default=4321)
    p.add_argument("--n", type=int, default=64)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--xpinn-epochs", type=int, default=800)
    p.add_argument("--qapinn-shallow-epochs", type=int, default=1200)
    p.add_argument("--qapinn-deep-epochs", type=int, default=1200)
    p.add_argument("--n-f", type=int, default=1024)
    p.add_argument("--n-b", type=int, default=256)
    p.add_argument("--n-interface", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--bc-weight", type=float, default=50.0)
    p.add_argument("--interface-weight", type=float, default=20.0)
    p.add_argument("--pinn-width", type=int, default=48)
    p.add_argument("--pinn-depth", type=int, default=3)
    p.add_argument("--xpinn-width", type=int, default=40)
    p.add_argument("--xpinn-depth", type=int, default=3)
    p.add_argument("--qapinn-qubits", type=int, default=3)
    p.add_argument("--qapinn-shallow-layers", type=int, default=1)
    p.add_argument("--qapinn-deep-layers", type=int, default=3)
    p.add_argument("--qapinn-shallow-hidden", type=int, default=24)
    p.add_argument("--qapinn-deep-hidden", type=int, default=48)
    p.add_argument("--print-every", type=int, default=100)
    p.add_argument("--skip-qapinn", action="store_true")
    p.add_argument("--quick", action="store_true")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.n = 64
        args.epochs = 80
        args.xpinn_epochs = 80
        args.qapinn_shallow_epochs = 100
        args.qapinn_deep_epochs = 100
        args.n_f = 256
        args.n_b = 96
        args.n_interface = 64
        args.pinn_width = 24
        args.xpinn_width = 24
        args.print_every = 40

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    x = np.linspace(0.0, 1.0, args.n)
    y = np.linspace(0.0, 1.0, args.n)
    xx, yy = np.meshgrid(x, y, indexing="ij")
    u_ref = exact_u(xx, yy)
    print(f"Using device: {device}")

    histories, models = {}, {}
    pinn = MLP(width=args.pinn_width, depth=args.pinn_depth)
    histories["PINN"] = train_pinn_like(pinn, "PINN", args, device)
    models["PINN"] = pinn
    torch.save(pinn.state_dict(), outdir / "models" / "poisson_pinn.pt")

    if not args.skip_qapinn:
        qapinn_shallow = QAPINN(
            n_qubits=args.qapinn_qubits,
            q_layers=args.qapinn_shallow_layers,
            hidden=args.qapinn_shallow_hidden,
            head_depth=1,
        )
        histories["QA-PINN-shallow"] = train_pinn_like(
            qapinn_shallow, "QA-PINN-shallow", args, device, epochs=args.qapinn_shallow_epochs
        )
        models["QA-PINN-shallow"] = qapinn_shallow
        torch.save(qapinn_shallow.state_dict(), outdir / "models" / "poisson_qapinn_shallow.pt")

        qapinn_deep = QAPINN(
            n_qubits=args.qapinn_qubits,
            q_layers=args.qapinn_deep_layers,
            hidden=args.qapinn_deep_hidden,
            head_depth=4,
        )
        histories["QA-PINN-deep"] = train_pinn_like(
            qapinn_deep, "QA-PINN-deep", args, device, epochs=args.qapinn_deep_epochs
        )
        models["QA-PINN-deep"] = qapinn_deep
        torch.save(qapinn_deep.state_dict(), outdir / "models" / "poisson_qapinn_deep.pt")

    xp_x, histories["XPINN-x"] = train_xpinn("x", args, device)
    models["XPINN-x"] = xp_x
    torch.save(xp_x.state_dict(), outdir / "models" / "poisson_xpinn_x.pt")

    xp_y, histories["XPINN-y"] = train_xpinn("y", args, device)
    models["XPINN-y"] = xp_y
    torch.save(xp_y.state_dict(), outdir / "models" / "poisson_xpinn_y.pt")

    predictions = {name: predict_grid(model, x, y, device) for name, model in models.items()}
    metrics = {name: grid_metrics(u_ref, pred) for name, pred in predictions.items()}
    for name in metrics:
        metrics[name]["params"] = count_parameters(models[name])
        metrics[name]["wall_time_sec"] = histories[name]["wall_time_sec"]
        metrics[name]["epochs"] = histories[name]["epochs"]
    metrics["_run"] = {
        "device": str(device),
        "n": args.n,
        "epochs": args.epochs,
        "xpinn_epochs": args.xpinn_epochs,
        "qapinn_shallow_epochs": args.qapinn_shallow_epochs,
        "qapinn_deep_epochs": args.qapinn_deep_epochs,
        "models": list(models.keys()),
        "equation": "u_xx + u_yy = -2*pi^2*sin(pi*x)*sin(pi*y), u=0 on boundary",
    }

    np.savez(
        outdir / "poisson_predictions.npz",
        x=x,
        y=y,
        u_ref=u_ref,
        **{name.replace("-", "_"): pred for name, pred in predictions.items()},
    )
    with open(outdir / "poisson_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / "poisson_histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)
    make_figures(outdir, x, y, u_ref, predictions, histories, {k: v for k, v in metrics.items() if not k.startswith("_")})
    print(json.dumps(metrics, indent=2))
    print(f"Done. Artifacts written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
