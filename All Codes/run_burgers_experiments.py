"""Run the full Burgers-equation comparison.

Outputs are written to:
  results/burgers_metrics.json
  results/burgers_predictions.npz
  results/figures/*.png
  results/models/*.pt

The defaults are intentionally CPU-friendly.  Increase --epochs and point
counts for final thesis-quality runs.
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

from burgers_models import MLP, QAPINN, XPINN, burgers_residual, set_seed, subnet_residual
from burgers_reference import BurgersDomain, grid_metrics, initial_condition, make_reference_grid


def tensor(x, device):
    return torch.as_tensor(x, dtype=torch.float32, device=device)


def count_parameters(model: torch.nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters() if p.requires_grad))


def sample_pinn_batch(n_f: int, n_i: int, n_b: int, domain: BurgersDomain, device):
    x_f = domain.x_min + (domain.x_max - domain.x_min) * torch.rand(n_f, 1, device=device)
    t_f = domain.t_min + (domain.t_max - domain.t_min) * torch.rand(n_f, 1, device=device)
    x_i = domain.x_min + (domain.x_max - domain.x_min) * torch.rand(n_i, 1, device=device)
    t_i = torch.zeros_like(x_i)
    u_i = -torch.sin(torch.pi * x_i)
    t_b = domain.t_min + (domain.t_max - domain.t_min) * torch.rand(n_b, 1, device=device)
    half = n_b // 2
    x_b = torch.cat([
        torch.full((half, 1), domain.x_min, device=device),
        torch.full((n_b - half, 1), domain.x_max, device=device),
    ])
    u_b = torch.zeros_like(x_b)
    return x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b


def train_pinn_like(model, label: str, args, domain: BurgersDomain, device, epochs: int | None = None):
    epochs = int(args.epochs if epochs is None else epochs)
    model.to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = {"loss_total": [], "loss_pde": [], "loss_ic": [], "loss_bc": []}
    start = time.perf_counter()
    for epoch in range(1, epochs + 1):
        batch = sample_pinn_batch(args.n_f, args.n_i, args.n_b, domain, device)
        x_f, t_f, x_i, t_i, u_i, x_b, t_b, u_b = batch
        f = burgers_residual(model, x_f, t_f, domain.nu)
        loss_pde = torch.mean(f**2)
        loss_ic = torch.mean((model(torch.cat([x_i, t_i], dim=1)) - u_i) ** 2)
        loss_bc = torch.mean((model(torch.cat([x_b, t_b], dim=1)) - u_b) ** 2)
        loss = loss_pde + args.data_weight * (loss_ic + loss_bc)

        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        for key, value in [
            ("loss_total", loss),
            ("loss_pde", loss_pde),
            ("loss_ic", loss_ic),
            ("loss_bc", loss_bc),
        ]:
            hist[key].append(float(value.detach().cpu()))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == epochs:
            print(f"{label:18s} epoch {epoch:5d}/{epochs} loss={hist['loss_total'][-1]:.3e}")
    hist["wall_time_sec"] = time.perf_counter() - start
    hist["epochs"] = epochs
    return hist


def xpinn_ranges(split: str, cuts: tuple[float, float], domain: BurgersDomain):
    c1, c2 = cuts
    if split == "time":
        return [
            ((domain.x_min, domain.x_max), (domain.t_min, c1)),
            ((domain.x_min, domain.x_max), (c1, c2)),
            ((domain.x_min, domain.x_max), (c2, domain.t_max)),
        ]
    return [
        ((domain.x_min, c1), (domain.t_min, domain.t_max)),
        ((c1, c2), (domain.t_min, domain.t_max)),
        ((c2, domain.x_max), (domain.t_min, domain.t_max)),
    ]


def sample_rect(n, xr, tr, device):
    x = xr[0] + (xr[1] - xr[0]) * torch.rand(n, 1, device=device)
    t = tr[0] + (tr[1] - tr[0]) * torch.rand(n, 1, device=device)
    return x, t


def train_xpinn(split: str, cuts: tuple[float, float], args, domain: BurgersDomain, device):
    model = XPINN(split=split, cuts=cuts, width=args.xpinn_width, depth=args.xpinn_depth).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)
    hist = {"loss_total": [], "loss_pde": [], "loss_data": [], "loss_interface": []}
    ranges = xpinn_ranges(split, cuts, domain)
    start = time.perf_counter()

    for epoch in range(1, args.xpinn_epochs + 1):
        loss_pde = torch.zeros((), device=device)
        loss_data = torch.zeros((), device=device)
        loss_interface = torch.zeros((), device=device)

        for k, (xr, tr) in enumerate(ranges):
            x_f, t_f = sample_rect(args.n_f // 3, xr, tr, device)
            f, _, _, _ = subnet_residual(model, k, x_f, t_f, domain.nu)
            loss_pde = loss_pde + torch.mean(f**2)

            n_i_local = max(8, args.n_i // 3)
            if split == "time" and k == 0:
                x_i = domain.x_min + (domain.x_max - domain.x_min) * torch.rand(n_i_local, 1, device=device)
                t_i = torch.zeros_like(x_i)
                u_i = -torch.sin(torch.pi * x_i)
                loss_data = loss_data + torch.mean((model.eval_subnet(k, x_i, t_i) - u_i) ** 2)
            if split == "space":
                x_i, t_i = sample_rect(n_i_local, xr, (domain.t_min, domain.t_min), device)
                u_i = -torch.sin(torch.pi * x_i)
                loss_data = loss_data + torch.mean((model.eval_subnet(k, x_i, t_i) - u_i) ** 2)

            n_b_local = max(8, args.n_b // 6)
            if split == "time" or (split == "space" and k == 0):
                t_b = tr[0] + (tr[1] - tr[0]) * torch.rand(n_b_local, 1, device=device)
                x_b = torch.full_like(t_b, domain.x_min)
                loss_data = loss_data + torch.mean(model.eval_subnet(k, x_b, t_b) ** 2)
            if split == "time" or (split == "space" and k == 2):
                t_b = tr[0] + (tr[1] - tr[0]) * torch.rand(n_b_local, 1, device=device)
                x_b = torch.full_like(t_b, domain.x_max)
                loss_data = loss_data + torch.mean(model.eval_subnet(k, x_b, t_b) ** 2)

        for j, cut in enumerate(cuts):
            if split == "time":
                x_int = domain.x_min + (domain.x_max - domain.x_min) * torch.rand(args.n_interface, 1, device=device)
                t_int = torch.full_like(x_int, cut)
            else:
                t_int = domain.t_min + (domain.t_max - domain.t_min) * torch.rand(args.n_interface, 1, device=device)
                x_int = torch.full_like(t_int, cut)
            f_l, u_l, ux_l, ut_l = subnet_residual(model, j, x_int, t_int, domain.nu)
            f_r, u_r, ux_r, ut_r = subnet_residual(model, j + 1, x_int, t_int, domain.nu)
            derivative_jump = ux_l - ux_r if split == "space" else ut_l - ut_r
            loss_interface = loss_interface + torch.mean((u_l - u_r) ** 2)
            loss_interface = loss_interface + 0.1 * torch.mean(derivative_jump**2)
            loss_interface = loss_interface + 0.1 * torch.mean((f_l - f_r) ** 2)

        loss = loss_pde + args.data_weight * loss_data + args.interface_weight * loss_interface
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()

        hist["loss_total"].append(float(loss.detach().cpu()))
        hist["loss_pde"].append(float(loss_pde.detach().cpu()))
        hist["loss_data"].append(float(loss_data.detach().cpu()))
        hist["loss_interface"].append(float(loss_interface.detach().cpu()))
        if epoch == 1 or epoch % args.print_every == 0 or epoch == args.xpinn_epochs:
            print(f"XPINN-{split:5s}       epoch {epoch:5d}/{args.xpinn_epochs} loss={hist['loss_total'][-1]:.3e}")

    hist["wall_time_sec"] = time.perf_counter() - start
    hist["epochs"] = int(args.xpinn_epochs)
    return model, hist


@torch.no_grad()
def predict_grid(model, x: np.ndarray, t: np.ndarray, device, batch_size: int = 8192) -> np.ndarray:
    xx, tt = np.meshgrid(x, t, indexing="ij")
    xt = np.stack([xx.ravel(), tt.ravel()], axis=1).astype(np.float32)
    outs = []
    model.eval()
    for i in range(0, len(xt), batch_size):
        pred = model(tensor(xt[i : i + batch_size], device)).cpu().numpy()
        outs.append(pred)
    return np.concatenate(outs, axis=0).reshape(len(x), len(t))


def save_figures(outdir: Path, x, t, u_ref, predictions, histories, metrics):
    figdir = outdir / "figures"
    figdir.mkdir(parents=True, exist_ok=True)
    tt, xx = np.meshgrid(t, x)

    for name, u_pred in predictions.items():
        fig, axes = plt.subplots(1, 3, figsize=(14, 3.6), constrained_layout=True)
        im0 = axes[0].pcolormesh(tt, xx, u_ref, shading="auto", cmap="seismic")
        axes[0].set_title("Reference")
        im1 = axes[1].pcolormesh(tt, xx, u_pred, shading="auto", cmap="seismic")
        axes[1].set_title(name)
        im2 = axes[2].pcolormesh(tt, xx, np.abs(u_pred - u_ref), shading="auto", cmap="magma")
        axes[2].set_title("|error|")
        for ax in axes:
            ax.set_xlabel("t")
            ax.set_ylabel("x")
        fig.colorbar(im0, ax=axes[:2], shrink=0.85)
        fig.colorbar(im2, ax=axes[2], shrink=0.85)
        fig.savefig(figdir / f"{name.lower().replace(' ', '_')}_field_error.png", dpi=220)
        plt.close(fig)

    slices = [0.25, 0.50, 0.75, 1.00]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    for ax, ts in zip(axes.ravel(), slices):
        j = int(np.argmin(np.abs(t - ts)))
        ax.plot(x, u_ref[:, j], "k-", lw=2.5, label="Reference")
        for name, u_pred in predictions.items():
            ax.plot(x, u_pred[:, j], lw=1.4, label=name)
        ax.set_title(f"t = {t[j]:.2f}")
        ax.set_xlabel("x")
        ax.set_ylabel("u")
        ax.grid(alpha=0.25)
    axes[0, 0].legend(fontsize=8, ncol=2)
    fig.savefig(figdir / "time_slice_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    for name, hist in histories.items():
        ax.semilogy(hist["loss_total"], label=name)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Total training loss")
    ax.set_title("Training loss comparison")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(figdir / "loss_comparison.png", dpi=220)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(8.5, 4.6), constrained_layout=True)
    labels = list(metrics)
    values = [metrics[k]["rel_l2"] for k in labels]
    ax.bar(labels, values, color=["#39568C", "#1F968B", "#73D055", "#DCE319", "#D65F5F"][: len(labels)])
    ax.set_ylabel("Relative L2 error")
    ax.set_title("Model error against finite-difference reference")
    ax.tick_params(axis="x", rotation=20)
    fig.savefig(figdir / "relative_l2_bar.png", dpi=220)
    plt.close(fig)

    metric_names = ["mse", "mae", "linf", "rel_l2"]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7), constrained_layout=True)
    labels = list(metrics)
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

    fig, ax = plt.subplots(figsize=(8.5, 4.8), constrained_layout=True)
    for name, u_pred in predictions.items():
        mse_t = np.mean((u_pred - u_ref) ** 2, axis=0)
        ax.semilogy(t, mse_t, label=name)
    ax.set_xlabel("t")
    ax.set_ylabel("MSE over x")
    ax.set_title("Burgers error by time")
    ax.grid(alpha=0.3)
    ax.legend()
    fig.savefig(figdir / "mse_vs_time.png", dpi=220)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7), constrained_layout=True)
    axes = axes.ravel()
    tt, xx = np.meshgrid(t, x)
    im = axes[0].pcolormesh(tt, xx, u_ref, shading="auto", cmap="seismic")
    axes[0].set_title("Reference")
    for ax, (name, pred) in zip(axes[1:], predictions.items()):
        ax.pcolormesh(tt, xx, pred, shading="auto", cmap="seismic")
        ax.set_title(name)
    for ax in axes:
        ax.set_xlabel("t")
        ax.set_ylabel("x")
    fig.colorbar(im, ax=axes.tolist(), shrink=0.8)
    fig.savefig(figdir / "prediction_montage.png", dpi=220)
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
    p.add_argument("--outdir", default="results")
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--nx", type=int, default=128)
    p.add_argument("--nt", type=int, default=81)
    p.add_argument("--epochs", type=int, default=800)
    p.add_argument("--xpinn-epochs", type=int, default=800)
    p.add_argument("--qapinn-shallow-epochs", type=int, default=1200)
    p.add_argument("--qapinn-deep-epochs", type=int, default=1200)
    p.add_argument("--n-f", type=int, default=768)
    p.add_argument("--n-i", type=int, default=192)
    p.add_argument("--n-b", type=int, default=192)
    p.add_argument("--n-interface", type=int, default=192)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--data-weight", type=float, default=30.0)
    p.add_argument("--interface-weight", type=float, default=15.0)
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
    p.add_argument("--quick", action="store_true", help="Use a tiny smoke-test configuration.")
    return p.parse_args()


def main():
    args = parse_args()
    if args.quick:
        args.epochs = 80
        args.xpinn_epochs = 80
        args.qapinn_shallow_epochs = 100
        args.qapinn_deep_epochs = 100
        args.n_f = 192
        args.n_i = 64
        args.n_b = 64
        args.n_interface = 64
        args.pinn_width = 24
        args.xpinn_width = 24
        args.print_every = 40

    outdir = Path(args.outdir)
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    domain = BurgersDomain()
    print(f"Using device: {device}")
    print("Generating finite-difference reference...")
    x, t, u_ref = make_reference_grid(nx=args.nx, nt=args.nt, domain=domain)

    histories = {}
    models = {}

    pinn = MLP(width=args.pinn_width, depth=args.pinn_depth)
    histories["PINN"] = train_pinn_like(pinn, "PINN", args, domain, device)
    models["PINN"] = pinn
    torch.save(pinn.state_dict(), outdir / "models" / "pinn_burgers.pt")

    if not args.skip_qapinn:
        qapinn_shallow = QAPINN(
            n_qubits=args.qapinn_qubits,
            q_layers=args.qapinn_shallow_layers,
            hidden=args.qapinn_shallow_hidden,
            head_depth=1,
        )
        histories["QA-PINN-shallow"] = train_pinn_like(
            qapinn_shallow, "QA-PINN-shallow", args, domain, device, epochs=args.qapinn_shallow_epochs
        )
        models["QA-PINN-shallow"] = qapinn_shallow
        torch.save(qapinn_shallow.state_dict(), outdir / "models" / "qapinn_shallow_burgers.pt")

        qapinn_deep = QAPINN(
            n_qubits=args.qapinn_qubits,
            q_layers=args.qapinn_deep_layers,
            hidden=args.qapinn_deep_hidden,
            head_depth=4,
        )
        histories["QA-PINN-deep"] = train_pinn_like(
            qapinn_deep, "QA-PINN-deep", args, domain, device, epochs=args.qapinn_deep_epochs
        )
        models["QA-PINN-deep"] = qapinn_deep
        torch.save(qapinn_deep.state_dict(), outdir / "models" / "qapinn_deep_burgers.pt")

    xpinn_time, histories["XPINN-time"] = train_xpinn("time", (1.0 / 3.0, 2.0 / 3.0), args, domain, device)
    models["XPINN-time"] = xpinn_time
    torch.save(xpinn_time.state_dict(), outdir / "models" / "xpinn_time_burgers.pt")

    xpinn_space, histories["XPINN-space"] = train_xpinn("space", (-0.2, 0.2), args, domain, device)
    models["XPINN-space"] = xpinn_space
    torch.save(xpinn_space.state_dict(), outdir / "models" / "xpinn_space_burgers.pt")

    predictions = {name: predict_grid(model, x, t, device) for name, model in models.items()}

    metrics = {name: grid_metrics(u_ref, pred) for name, pred in predictions.items()}
    for name in metrics:
        metrics[name]["params"] = count_parameters(models[name])
        metrics[name]["wall_time_sec"] = histories[name]["wall_time_sec"]
        metrics[name]["epochs"] = histories[name]["epochs"]
    metrics["_run"] = {
        "device": str(device),
        "nu": domain.nu,
        "nx": args.nx,
        "nt": args.nt,
        "epochs": args.epochs,
        "xpinn_epochs": args.xpinn_epochs,
        "qapinn_shallow_epochs": args.qapinn_shallow_epochs,
        "qapinn_deep_epochs": args.qapinn_deep_epochs,
        "models": list(models.keys()),
    }

    np.savez(
        outdir / "burgers_predictions.npz",
        x=x,
        t=t,
        u_ref=u_ref,
        **{name.replace("-", "_").replace(" ", "_"): pred for name, pred in predictions.items()},
    )
    with open(outdir / "burgers_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    with open(outdir / "burgers_histories.json", "w", encoding="utf-8") as f:
        json.dump(histories, f, indent=2)

    save_figures(outdir, x, t, u_ref, predictions, histories, {k: v for k, v in metrics.items() if not k.startswith("_")})
    print(json.dumps(metrics, indent=2))
    print(f"Done. Artifacts written to {outdir.resolve()}")


if __name__ == "__main__":
    main()
