"""
Generate paper-style figures from cached results.
"""

import json
import os
from typing import Dict, List, Optional

import matplotlib.pyplot as plt


def _load_json(path: str) -> Optional[Dict]:
    if not os.path.exists(path):
        print(f"Missing: {path}")
        return None
    with open(path, "r") as f:
        return json.load(f)


def plot_fc_variants(results_path: str, save_dir: str) -> None:
    data = _load_json(results_path)
    if not data:
        return
    configs = data.get("configs", [])
    if not configs:
        return

    xs = [c["total_params"] for c in configs]
    ys = [c["d_int90_mean"] for c in configs]
    yerrs = [c.get("d_int90_std", 0.0) for c in configs]
    sizes = [c["width"] * 2 for c in configs]
    depths = [c["depth"] for c in configs]

    plt.figure(figsize=(10, 6))
    for x, y, yerr in zip(xs, ys, yerrs):
        plt.errorbar(x, y, yerr=yerr, fmt="none", ecolor="gray", alpha=0.6)
    sc = plt.scatter(xs, ys, s=sizes, c=depths, cmap="viridis", alpha=0.8, edgecolors="black")
    cbar = plt.colorbar(sc)
    cbar.set_label("Depth")
    plt.xscale("log")
    plt.xlabel("Number of parameters D")
    plt.ylabel("Intrinsic dimension $d_{int90}$")
    plt.title("Figure 3: MNIST FC Variants")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    out = os.path.join(save_dir, "paper_fig_3_fc_variants.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def _extract_subspace_curve(data: Dict) -> Dict[str, List[float]]:
    dims = data.get("dimensions_tested") or data.get("dimensions")
    accs = data.get("accuracies") or data.get("accuracies_mean")
    accs_std = data.get("accuracies_std")
    return {"dims": dims, "accs": accs, "accs_std": accs_std}


def plot_accuracy_vs_d(fc_path: str, lenet_path: str, save_dir: str) -> None:
    fc = _load_json(fc_path)
    ln = _load_json(lenet_path)
    if not fc or not ln:
        return

    fc_curve = _extract_subspace_curve(fc)
    ln_curve = _extract_subspace_curve(ln)

    plt.figure(figsize=(10, 6))
    if fc_curve["accs_std"]:
        plt.errorbar(fc_curve["dims"], fc_curve["accs"], yerr=fc_curve["accs_std"],
                     fmt="o-", label="FC")
    else:
        plt.plot(fc_curve["dims"], fc_curve["accs"], "o-", label="FC")
    if ln_curve["accs_std"]:
        plt.errorbar(ln_curve["dims"], ln_curve["accs"], yerr=ln_curve["accs_std"],
                     fmt="o-", label="LeNet")
    else:
        plt.plot(ln_curve["dims"], ln_curve["accs"], "o-", label="LeNet")

    plt.xscale("log")
    plt.xlabel("Subspace dimension (d)")
    plt.ylabel("Test accuracy (%)")
    plt.title("Figure S6: Accuracy vs Subspace Dimension")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    out = os.path.join(save_dir, "paper_fig_s6_accuracy_vs_d.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def plot_direct_vs_subspace(
    direct_fc_path: str,
    direct_lenet_path: str,
    subspace_fc_path: str,
    subspace_lenet_path: str,
    save_dir: str
) -> None:
    direct_fc = _load_json(direct_fc_path)
    direct_ln = _load_json(direct_lenet_path)
    sub_fc = _load_json(subspace_fc_path)
    sub_ln = _load_json(subspace_lenet_path)
    if not direct_fc or not direct_ln:
        return
    if not sub_fc or not sub_ln:
        print("Missing subspace results for Figure 4")
        return

    sub_fc_curve = _extract_subspace_curve(sub_fc)
    sub_ln_curve = _extract_subspace_curve(sub_ln)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    # FC
    ax1 = axes[0]
    ax1.scatter(
        [m["total_params"] for m in direct_fc.get("models", [])],
        [m["acc"] for m in direct_fc.get("models", [])],
        c="gray", alpha=0.6, label="Direct"
    )
    ax1.scatter(sub_fc_curve["dims"], sub_fc_curve["accs"], c="#4C72B0", label="Subspace")
    ax1.set_xscale("log")
    ax1.set_xlabel("Number of trainable parameters")
    ax1.set_ylabel("Validation accuracy")
    ax1.set_title("FC Networks (MNIST)")
    ax1.grid(True, alpha=0.3)
    ax1.legend()

    # Conv
    ax2 = axes[1]
    ax2.scatter(
        [m["total_params"] for m in direct_ln.get("models", [])],
        [m["acc"] for m in direct_ln.get("models", [])],
        c="gray", alpha=0.6, label="Direct"
    )
    ax2.scatter(sub_ln_curve["dims"], sub_ln_curve["accs"], c="#4C72B0", label="Subspace")
    ax2.set_xscale("log")
    ax2.set_xlabel("Number of trainable parameters")
    ax2.set_ylabel("Validation accuracy")
    ax2.set_title("Conv Networks (MNIST)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    plt.tight_layout()
    out = os.path.join(save_dir, "paper_fig_4_direct_vs_subspace.png")
    plt.savefig(out, dpi=150, bbox_inches="tight")
    print(f"Saved {out}")
    plt.close()


def main(save_dir: str = "results") -> None:
    os.makedirs(save_dir, exist_ok=True)

    plot_fc_variants(os.path.join(save_dir, "mnist_fc_variants.json"), save_dir)
    plot_direct_vs_subspace(
        os.path.join(save_dir, "mnist_direct_fc.json"),
        os.path.join(save_dir, "mnist_direct_lenet.json"),
        os.path.join(save_dir, "mnist_fc_results.json"),
        os.path.join(save_dir, "mnist_lenet_results.json"),
        save_dir
    )
    plot_accuracy_vs_d(
        os.path.join(save_dir, "mnist_fc_results.json"),
        os.path.join(save_dir, "mnist_lenet_results.json"),
        save_dir
    )


if __name__ == "__main__":
    main()
