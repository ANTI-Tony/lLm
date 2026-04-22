"""
Train the input-conditioned depth predictor.

Takes the .pt produced by build_predictor_data.py and trains a small MLP
that maps features -> oracle depth class. Reports:
  * classification accuracy
  * mean absolute depth error (MAE in ns-index space)
  * confusion matrix over depth classes
  * feature importance (via permutation)

This is a first pass — Phase 2 will iterate with richer features
(CLIP-text embedding, topic classifier, etc).

Usage:
    python train_predictor.py --data data/predictor/train.pt \\
        --output checkpoints/predictor_v1.pt
"""

import argparse
import copy
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset, random_split


class DepthPredictor(nn.Module):
    def __init__(self, in_dim: int, n_classes: int, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.GELU(),
            nn.Linear(hidden, hidden),
            nn.GELU(),
            nn.Linear(hidden, n_classes),
        )

    def forward(self, x):
        return self.net(x)


def evaluate(model, loader, device):
    model.eval()
    correct = 0
    total = 0
    abs_err = 0.0
    conf = None
    all_pred, all_true = [], []
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            pred = logits.argmax(dim=-1)
            correct += (pred == y).sum().item()
            total += y.size(0)
            abs_err += (pred - y).abs().float().sum().item()
            all_pred.append(pred.cpu())
            all_true.append(y.cpu())
    all_pred = torch.cat(all_pred).numpy()
    all_true = torch.cat(all_true).numpy()
    return {"accuracy": correct / total, "mae": abs_err / total,
            "pred": all_pred, "true": all_true}


def permutation_importance(model, X, y, baseline_acc, device, n_repeats=5):
    """Shuffle one feature at a time, measure accuracy drop."""
    model.eval()
    n, D = X.shape
    drops = []
    for d in range(D):
        accs = []
        for _ in range(n_repeats):
            X_p = X.clone()
            perm = torch.randperm(n)
            X_p[:, d] = X[perm, d]
            with torch.no_grad():
                pred = model(X_p.to(device)).argmax(-1).cpu()
            accs.append((pred == y).float().mean().item())
        drops.append(baseline_acc - np.mean(accs))
    return drops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", default="checkpoints/predictor_v1.pt")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--hidden", type=int, default=128)
    parser.add_argument("--val_frac", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    data = torch.load(args.data)
    X, y = data["X"], data["y"]
    ns_values = data["ns_values"]
    n_classes = len(ns_values) + 1  # +1 for "never" sentinel

    # Standardize features (per-column).
    mean = X.mean(0, keepdim=True)
    std = X.std(0, keepdim=True).clamp(min=1e-4)
    X_n = (X - mean) / std

    # Split.
    n = X_n.size(0)
    n_val = int(n * args.val_frac)
    n_train = n - n_val
    full = TensorDataset(X_n, y)
    train_ds, val_ds = random_split(full, [n_train, n_val],
                                    generator=torch.Generator().manual_seed(args.seed))
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,
                              shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = DepthPredictor(X_n.size(1), n_classes, args.hidden).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_val = 0.0
    best_state = None
    for epoch in range(args.epochs):
        model.train()
        running = 0.0
        for x, y_b in train_loader:
            x, y_b = x.to(device), y_b.to(device)
            logits = model(x)
            loss = F.cross_entropy(logits, y_b)
            opt.zero_grad()
            loss.backward()
            opt.step()
            running += loss.item()

        val = evaluate(model, val_loader, device)
        train_loss = running / len(train_loader)
        if epoch % 5 == 0 or epoch == args.epochs - 1:
            print(f"epoch {epoch:3d}  train_loss={train_loss:.3f}  "
                  f"val_acc={val['accuracy']:.3f}  val_mae={val['mae']:.3f}")
        if val["accuracy"] > best_val:
            best_val = val["accuracy"]
            best_state = copy.deepcopy(model.state_dict())

    model.load_state_dict(best_state)

    # Final eval + importance.
    final = evaluate(model, val_loader, device)
    print(f"\n=== Best Validation ===")
    print(f"accuracy: {final['accuracy']:.3f}")
    print(f"mae (classes): {final['mae']:.3f}")

    # Baselines for context.
    majority_class = int(torch.mode(y).values)
    majority_acc = (y == majority_class).float().mean().item()
    print(f"\nmajority-class baseline: {majority_acc:.3f} "
          f"(class={majority_class}, ns={ns_values[majority_class] if majority_class < len(ns_values) else 'never'})")

    # Confusion matrix.
    print("\nconfusion (rows=true, cols=pred):")
    C = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(final["true"], final["pred"]):
        C[t, p] += 1
    header = " " * 8 + "".join(f"{('ns=' + str(ns)):>8}" for ns in ns_values) + f"{'never':>8}"
    print(header)
    for i in range(n_classes):
        name = f"ns={ns_values[i]}" if i < len(ns_values) else "never"
        row = "".join(f"{C[i, j]:>8d}" for j in range(n_classes))
        print(f"{name:>8}{row}")

    # Feature importance via permutation.
    print("\npermutation feature importance (drop in val acc):")
    X_val = torch.stack([v[0] for v in val_ds])
    y_val = torch.stack([v[1] for v in val_ds])
    drops = permutation_importance(model, X_val, y_val,
                                   baseline_acc=final["accuracy"],
                                   device=device)
    for d, name in zip(drops, FEATURE_NAMES):
        print(f"  {name:<20s} {d:+.4f}")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(),
                "mean": mean, "std": std,
                "ns_values": ns_values,
                "feature_dim": X_n.size(1),
                "n_classes": n_classes,
                "val_accuracy": final["accuracy"],
                "val_mae": final["mae"],
                "confusion": C.tolist(),
                "feature_importance": list(zip(FEATURE_NAMES, drops))},
               args.output)
    print(f"\n[info] saved -> {args.output}")


FEATURE_NAMES = (["n_chars", "n_words", "n_sents",
                  "n_nums", "op+", "op-", "op*", "op/", "op=", "op^",
                  "n_parens", "has_frac"]
                 + ["kw_if", "kw_then", "kw_because", "kw_cause",
                    "kw_how_many", "kw_what_is", "kw_find", "kw_solve",
                    "kw_when", "kw_where", "kw_compare", "kw_total",
                    "kw_average", "kw_ratio", "kw_percent", "kw_fraction"])


if __name__ == "__main__":
    main()
