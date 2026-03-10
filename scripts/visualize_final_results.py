#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import html
import json
import math
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Sequence

BG = "#ffffff"
AXIS = "#111827"
GRID = "#d1d5db"
TEXT = "#111827"
SUBTLE = "#6b7280"
MODE_COLORS = {"id_module": "#0f766e", "lora": "#b45309"}
STAGE_COLORS = {1: "#c2410c", 2: "#2563eb", 3: "#047857"}
SEED_COLORS = {42: "#1d4ed8", 43: "#dc2626", 44: "#7c3aed"}
MODE_LABELS = {"id_module": "Intrinsic Dim", "lora": "LoRA"}
FONT = "'Times New Roman', Georgia, serif"
SANS = "Arial, Helvetica, sans-serif"


@dataclass
class RunRecord:
    mode: str
    seed: int
    dataset: str
    model_name: str
    path: Path
    run_name: str
    best_test_acc: float
    final_test_acc: float
    final_train_acc: float
    epochs_trained: int
    trainable_params: int
    total_params: int
    total_time_sec: float
    lr: float
    weight_decay: float
    grad_accum_steps: int
    no_amp: bool
    train_acc_history: List[float]
    test_acc_history: List[float]
    train_loss_history: List[float]
    test_loss_history: List[float]
    extra: Dict[str, object]

    @property
    def total_time_hours(self) -> float:
        return self.total_time_sec / 3600.0

    @property
    def generalization_gap(self) -> float:
        return self.final_train_acc - self.final_test_acc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate pure-SVG final figures for the ViT intrinsic-dim vs LoRA study.")
    parser.add_argument("--results_root", type=Path, default=Path("results/vit_intrinsic/cifar100"))
    parser.add_argument("--tuning_summary", type=Path, default=Path("results/vit_intrinsic_tuning/tuning_summary__cifar100__vit_base_patch16_224__seed42.json"))
    parser.add_argument("--output_dir", type=Path, default=Path("docs/figures/final_analysis"))
    parser.add_argument("--dataset", default="cifar100")
    parser.add_argument("--model_name", default="vit_base_patch16_224")
    parser.add_argument("--seeds", default="42,43")
    parser.add_argument("--id_subspace_dim", type=int, default=666724)
    parser.add_argument("--lora_rank", type=int, default=8)
    return parser.parse_args()


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def mean(values: Sequence[float]) -> float:
    return statistics.fmean(values) if values else 0.0


def std(values: Sequence[float]) -> float:
    return statistics.pstdev(values) if len(values) > 1 else 0.0


def hours(seconds: float) -> float:
    return seconds / 3600.0


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def fmt_num(value: float, digits: int = 2) -> str:
    return f"{value:.{digits}f}"


def load_records(results_root: Path) -> List[RunRecord]:
    records: List[RunRecord] = []
    for path in sorted(results_root.glob("*/*.json")):
        if path.name.startswith("tune__"):
            continue
        payload = load_json(path)
        args = payload.get("args", {})
        metrics = payload.get("metrics", {})
        if not metrics:
            continue
        records.append(
            RunRecord(
                mode=str(args.get("mode", "")),
                seed=int(args.get("seed", -1)),
                dataset=str(args.get("dataset", "")),
                model_name=str(args.get("model_name", "")),
                path=path,
                run_name=str(payload.get("run_name", path.stem)),
                best_test_acc=float(metrics.get("best_test_acc", 0.0)),
                final_test_acc=float(metrics.get("final_test_acc", 0.0)),
                final_train_acc=float(metrics.get("final_train_acc", 0.0)),
                epochs_trained=int(metrics.get("epochs_trained", 0)),
                trainable_params=int(metrics.get("trainable_params", 0)),
                total_params=int(metrics.get("total_params", 0)),
                total_time_sec=float(metrics.get("total_time_sec", 0.0)),
                lr=float(args.get("lr", 0.0)),
                weight_decay=float(args.get("weight_decay", 0.0)),
                grad_accum_steps=int(args.get("grad_accum_steps", 1)),
                no_amp=bool(args.get("no_amp", False)),
                train_acc_history=list(metrics.get("train_acc_history", [])),
                test_acc_history=list(metrics.get("test_acc_history", [])),
                train_loss_history=list(metrics.get("train_loss_history", [])),
                test_loss_history=list(metrics.get("test_loss_history", [])),
                extra={
                    "lora_rank": args.get("lora_rank"),
                    "lora_alpha": args.get("lora_alpha"),
                    "lora_scope": args.get("lora_scope"),
                    "subspace_dim": args.get("subspace_dim"),
                    "projection": args.get("projection"),
                    "id_scope": args.get("id_scope"),
                },
            )
        )
    return records


def select_final_records(records: List[RunRecord], dataset: str, model_name: str, seeds: Iterable[int], id_subspace_dim: int, lora_rank: int) -> List[RunRecord]:
    selected: List[RunRecord] = []
    seed_set = set(seeds)
    for record in records:
        if record.dataset != dataset or record.model_name != model_name or record.seed not in seed_set:
            continue
        if record.epochs_trained < 20:
            continue
        if record.mode == "id_module" and int(record.extra.get("subspace_dim") or -1) == id_subspace_dim and record.grad_accum_steps == 4:
            selected.append(record)
        elif record.mode == "lora" and int(record.extra.get("lora_rank") or -1) == lora_rank and record.grad_accum_steps == 4:
            selected.append(record)
    selected.sort(key=lambda r: (r.mode, r.seed))
    return selected


def select_dimension_records(records: List[RunRecord], dataset: str, model_name: str, target_dim: int) -> List[RunRecord]:
    chosen: List[RunRecord] = []
    for record in records:
        if record.dataset != dataset or record.model_name != model_name or record.mode != "id_module":
            continue
        dim = int(record.extra.get("subspace_dim") or -1)
        if dim in (371812, target_dim):
            if dim == 371812 or record.seed in (42, 43):
                chosen.append(record)
    chosen.sort(key=lambda r: (int(r.extra.get("subspace_dim") or 0), r.seed))
    return chosen


def summarize(records: List[RunRecord]) -> Dict[str, Dict[str, float]]:
    out: Dict[str, Dict[str, float]] = {}
    for mode in sorted({r.mode for r in records}):
        subset = [r for r in records if r.mode == mode]
        out[mode] = {
            "n": float(len(subset)),
            "best_mean": mean([r.best_test_acc for r in subset]),
            "best_std": std([r.best_test_acc for r in subset]),
            "final_mean": mean([r.final_test_acc for r in subset]),
            "final_std": std([r.final_test_acc for r in subset]),
            "train_mean": mean([r.final_train_acc for r in subset]),
            "gap_mean": mean([r.generalization_gap for r in subset]),
            "trainable_mean": mean([float(r.trainable_params) for r in subset]),
        }
    return out


class Svg:
    def __init__(self, width: int, height: int, title: str = ""):
        self.width = width
        self.height = height
        self.parts: List[str] = [
            f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{html.escape(title)}">',
            f'<rect x="0" y="0" width="{width}" height="{height}" fill="{BG}" />',
        ]

    def add(self, fragment: str) -> None:
        self.parts.append(fragment)

    def line(self, x1: float, y1: float, x2: float, y2: float, stroke: str = AXIS, stroke_width: float = 1.0, dash: str | None = None, opacity: float = 1.0) -> None:
        attrs = [
            f'x1="{x1:.2f}"', f'y1="{y1:.2f}"', f'x2="{x2:.2f}"', f'y2="{y2:.2f}"',
            f'stroke="{stroke}"', f'stroke-width="{stroke_width}"', f'opacity="{opacity}"'
        ]
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        self.add(f'<line {" ".join(attrs)} />')

    def rect(self, x: float, y: float, width: float, height: float, fill: str, stroke: str = "none", stroke_width: float = 0.0, rx: float = 0.0, opacity: float = 1.0) -> None:
        self.add(
            f'<rect x="{x:.2f}" y="{y:.2f}" width="{width:.2f}" height="{height:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" rx="{rx}" opacity="{opacity}" />'
        )

    def circle(self, cx: float, cy: float, r: float, fill: str, stroke: str = "none", stroke_width: float = 0.0) -> None:
        self.add(f'<circle cx="{cx:.2f}" cy="{cy:.2f}" r="{r:.2f}" fill="{fill}" stroke="{stroke}" stroke-width="{stroke_width}" />')

    def polyline(self, points: Sequence[tuple[float, float]], stroke: str, stroke_width: float = 2.0, fill: str = "none", dash: str | None = None, opacity: float = 1.0) -> None:
        joined = " ".join(f"{x:.2f},{y:.2f}" for x, y in points)
        attrs = [f'points="{joined}"', f'stroke="{stroke}"', f'stroke-width="{stroke_width}"', f'fill="{fill}"', f'opacity="{opacity}"']
        if dash:
            attrs.append(f'stroke-dasharray="{dash}"')
        attrs.append('stroke-linecap="round"')
        attrs.append('stroke-linejoin="round"')
        self.add(f'<polyline {" ".join(attrs)} />')

    def text(self, x: float, y: float, text: str, size: int = 12, anchor: str = "middle", fill: str = TEXT, weight: str = "normal", rotate: float | None = None, family: str = FONT) -> None:
        transform = f' transform="rotate({rotate:.2f} {x:.2f} {y:.2f})"' if rotate is not None else ""
        self.add(
            f'<text x="{x:.2f}" y="{y:.2f}" font-size="{size}" text-anchor="{anchor}" fill="{fill}" font-family="{family}" font-weight="{weight}"{transform}>{html.escape(text)}</text>'
        )

    def save(self, path: Path) -> None:
        path.write_text("\n".join(self.parts + ["</svg>"]) + "\n", encoding="utf-8")


@dataclass
class PlotArea:
    x: float
    y: float
    width: float
    height: float
    xmin: float
    xmax: float
    ymin: float
    ymax: float

    def px(self, value: float) -> float:
        if self.xmax == self.xmin:
            return self.x
        return self.x + (value - self.xmin) / (self.xmax - self.xmin) * self.width

    def py(self, value: float) -> float:
        if self.ymax == self.ymin:
            return self.y + self.height
        return self.y + self.height - (value - self.ymin) / (self.ymax - self.ymin) * self.height


def nice_step(raw_step: float) -> float:
    if raw_step <= 0:
        return 1.0
    exponent = math.floor(math.log10(raw_step))
    fraction = raw_step / (10 ** exponent)
    if fraction <= 1:
        nice_fraction = 1
    elif fraction <= 2:
        nice_fraction = 2
    elif fraction <= 5:
        nice_fraction = 5
    else:
        nice_fraction = 10
    return nice_fraction * (10 ** exponent)


def nice_ticks(vmin: float, vmax: float, count: int = 5) -> List[float]:
    if vmin == vmax:
        return [vmin]
    raw_step = (vmax - vmin) / max(count - 1, 1)
    step = nice_step(raw_step)
    start = math.floor(vmin / step) * step
    end = math.ceil(vmax / step) * step
    ticks: List[float] = []
    value = start
    while value <= end + step * 0.5:
        ticks.append(round(value, 8))
        value += step
    return ticks


def pad_range(values: Sequence[float], pad_ratio: float = 0.06) -> tuple[float, float]:
    low = min(values)
    high = max(values)
    if low == high:
        return low - 1.0, high + 1.0
    pad = (high - low) * pad_ratio
    return low - pad, high + pad


def draw_plot_frame(svg: Svg, plot: PlotArea, title: str, x_label: str, y_label: str, x_ticks: Sequence[float], y_ticks: Sequence[float], x_formatter: Callable[[float], str], y_formatter: Callable[[float], str], x_tick_rotation: float = 0.0) -> None:
    svg.rect(plot.x, plot.y, plot.width, plot.height, fill="#ffffff", stroke="#9ca3af", stroke_width=1.0)
    for tick in y_ticks:
        y = plot.py(tick)
        svg.line(plot.x, y, plot.x + plot.width, y, stroke=GRID, stroke_width=0.9, dash="4 4")
        svg.text(plot.x - 10, y + 4, y_formatter(tick), size=11, anchor="end", fill=SUBTLE, family=SANS)
    for tick in x_ticks:
        x = plot.px(tick)
        svg.line(x, plot.y, x, plot.y + plot.height, stroke=GRID, stroke_width=0.8, dash="4 4", opacity=0.55)
        label_y = plot.y + plot.height + (24 if x_tick_rotation else 18)
        svg.text(x, label_y, x_formatter(tick), size=11, anchor="middle", fill=SUBTLE, rotate=x_tick_rotation if x_tick_rotation else None, family=SANS)
    svg.line(plot.x, plot.y + plot.height, plot.x + plot.width, plot.y + plot.height, stroke=AXIS, stroke_width=1.2)
    svg.line(plot.x, plot.y, plot.x, plot.y + plot.height, stroke=AXIS, stroke_width=1.2)
    svg.text(plot.x + plot.width / 2, plot.y - 18, title, size=16, weight="bold")
    svg.text(plot.x + plot.width / 2, plot.y + plot.height + 42, x_label, size=12, family=SANS)
    svg.text(plot.x - 54, plot.y + plot.height / 2, y_label, size=12, rotate=-90, family=SANS)


def legend_box(svg: Svg, x: float, y: float, items: Sequence[tuple[str, str, str]]) -> None:
    height = 22 * len(items) + 14
    width = 180
    svg.rect(x, y, width, height, fill="#ffffff", stroke="#d1d5db", stroke_width=0.8, rx=8)
    cy = y + 18
    for kind, color, label in items:
        if kind == "bar":
            svg.rect(x + 12, cy - 8, 14, 10, fill=color, stroke=AXIS, stroke_width=0.6)
        elif kind == "line":
            svg.line(x + 12, cy - 3, x + 26, cy - 3, stroke=color, stroke_width=2.4)
        elif kind == "dash":
            svg.line(x + 12, cy - 3, x + 26, cy - 3, stroke=color, stroke_width=2.0, dash="6 4")
        elif kind == "dot":
            svg.circle(x + 19, cy - 3, 4.0, fill=color, stroke=AXIS, stroke_width=0.6)
        svg.text(x + 34, cy, label, size=11, anchor="start", fill=TEXT, family=SANS)
        cy += 22


def write_csvs(records: List[RunRecord], summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    with (output_dir / "final_runs.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "mode", "seed", "best_test_acc", "final_test_acc", "final_train_acc", "generalization_gap",
            "epochs_trained", "trainable_params", "total_params", "total_time_sec", "total_time_hours",
            "lr", "weight_decay", "grad_accum_steps", "no_amp", "json_path"
        ])
        for r in records:
            writer.writerow([
                r.mode, r.seed, fmt_num(r.best_test_acc, 4), fmt_num(r.final_test_acc, 4), fmt_num(r.final_train_acc, 4),
                fmt_num(r.generalization_gap, 4), r.epochs_trained, r.trainable_params, r.total_params,
                fmt_num(r.total_time_sec, 2), fmt_num(r.total_time_hours, 4), r.lr, r.weight_decay,
                r.grad_accum_steps, int(r.no_amp), str(r.path)
            ])
    with (output_dir / "mode_summary.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["mode", "n", "best_mean", "best_std", "final_mean", "final_std", "train_mean", "gap_mean", "trainable_mean"])
        for mode, values in summary.items():
            writer.writerow([
                mode,
                int(values["n"]),
                fmt_num(values["best_mean"], 4),
                fmt_num(values["best_std"], 4),
                fmt_num(values["final_mean"], 4),
                fmt_num(values["final_std"], 4),
                fmt_num(values["train_mean"], 4),
                fmt_num(values["gap_mean"], 4),
                fmt_num(values["trainable_mean"], 1),
            ])


def save_accuracy_by_seed(records: List[RunRecord], output_dir: Path) -> None:
    seeds = sorted({r.seed for r in records})
    modes = ["id_module", "lora"]
    svg = Svg(860, 520, "Final CIFAR-100 Accuracy by Seed")
    plot = PlotArea(90, 70, 690, 320, -0.5, len(seeds) - 0.5, 91.6, 92.3)
    x_ticks = list(range(len(seeds)))
    y_ticks = [91.6, 91.8, 92.0, 92.2]
    draw_plot_frame(svg, plot, "Final CIFAR-100 Accuracy by Seed", "Seed", "Accuracy (%)", x_ticks, y_ticks, lambda t: str(seeds[int(round(t))]), lambda t: f"{t:.1f}")
    bar_width = 0.28
    offsets = {"id_module": -0.16, "lora": 0.16}
    for mode in modes:
        for idx, seed in enumerate(seeds):
            record = next(r for r in records if r.mode == mode and r.seed == seed)
            x_center = idx + offsets[mode]
            x_left = plot.px(x_center - bar_width / 2)
            x_right = plot.px(x_center + bar_width / 2)
            y_top = plot.py(record.final_test_acc)
            y_base = plot.py(plot.ymin)
            svg.rect(x_left, y_top, x_right - x_left, y_base - y_top, fill=MODE_COLORS[mode], stroke=AXIS, stroke_width=0.7)
            svg.circle((x_left + x_right) / 2, plot.py(record.best_test_acc), 4.4, fill="#111111", stroke="#ffffff", stroke_width=1.0)
            svg.text((x_left + x_right) / 2, y_top - 8, f"{record.final_test_acc:.2f}", size=10, family=SANS)
    legend_box(svg, 618, 86, [("bar", MODE_COLORS["id_module"], "Intrinsic Dim final"), ("bar", MODE_COLORS["lora"], "LoRA final"), ("dot", "#111111", "Best checkpoint")])
    svg.save(output_dir / "final_accuracy_by_seed.svg")


def draw_error_bar(svg: Svg, x: float, y_mid: float, y_low: float, y_high: float) -> None:
    svg.line(x, y_low, x, y_high, stroke=AXIS, stroke_width=1.1)
    svg.line(x - 6, y_low, x + 6, y_low, stroke=AXIS, stroke_width=1.1)
    svg.line(x - 6, y_high, x + 6, y_high, stroke=AXIS, stroke_width=1.1)


def save_mean_accuracy_and_gap(summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    modes = ["id_module", "lora"]
    svg = Svg(980, 440, "Mean Accuracy and Generalization Gap")
    left = PlotArea(80, 80, 360, 260, -0.5, 1.5, 91.6, 92.2)
    right = PlotArea(560, 80, 320, 260, -0.5, 1.5, 0.0, 9.0)
    draw_plot_frame(svg, left, "Mean Final Accuracy", "Method", "Accuracy (%)", [0, 1], [91.6, 91.8, 92.0, 92.2], lambda t: MODE_LABELS[modes[int(round(t))]], lambda t: f"{t:.1f}")
    draw_plot_frame(svg, right, "Generalization Gap", "Method", "Train - test gap (%)", [0, 1], [0, 2, 4, 6, 8], lambda t: MODE_LABELS[modes[int(round(t))]], lambda t: f"{t:.0f}")
    for idx, mode in enumerate(modes):
        x_left = left.px(idx - 0.18)
        x_right = left.px(idx + 0.18)
        y_top = left.py(summary[mode]["final_mean"])
        svg.rect(x_left, y_top, x_right - x_left, left.py(left.ymin) - y_top, fill=MODE_COLORS[mode], stroke=AXIS, stroke_width=0.7)
        mid_x = (x_left + x_right) / 2
        draw_error_bar(svg, mid_x, left.py(summary[mode]["final_mean"]), left.py(summary[mode]["final_mean"] - summary[mode]["final_std"]), left.py(summary[mode]["final_mean"] + summary[mode]["final_std"]))
        svg.text(mid_x, y_top - 8, f"{summary[mode]['final_mean']:.2f}", size=10, family=SANS)
        x_left_r = right.px(idx - 0.18)
        x_right_r = right.px(idx + 0.18)
        y_top_r = right.py(summary[mode]["gap_mean"])
        svg.rect(x_left_r, y_top_r, x_right_r - x_left_r, right.py(right.ymin) - y_top_r, fill=MODE_COLORS[mode], stroke=AXIS, stroke_width=0.7)
        svg.text((x_left_r + x_right_r) / 2, y_top_r - 8, f"{summary[mode]['gap_mean']:.2f}", size=10, family=SANS)
    svg.save(output_dir / "mean_accuracy_and_gap.svg")


def save_runtime_tradeoff(records: List[RunRecord], output_dir: Path) -> None:
    fair = sorted([r for r in records if r.seed == 43], key=lambda r: r.mode)
    svg = Svg(1000, 440, "Seed-43 Runtime Tradeoff")
    left = PlotArea(80, 80, 360, 260, -0.5, 1.5, 0.0, 4.5)
    right = PlotArea(580, 80, 320, 260, 0.0, 4.5, 91.8, 92.1)
    draw_plot_frame(svg, left, "Runtime on the Same 4090 Host", "Method", "Runtime (hours)", [0, 1], [0, 1, 2, 3, 4], lambda t: MODE_LABELS[["id_module", "lora"][int(round(t))]], lambda t: f"{t:.0f}")
    draw_plot_frame(svg, right, "Accuracy vs Runtime", "Runtime (hours)", "Final accuracy (%)", [0, 1, 2, 3, 4], [91.8, 91.9, 92.0, 92.1], lambda t: f"{t:.0f}", lambda t: f"{t:.1f}")
    for idx, record in enumerate(fair):
        x_left = left.px(idx - 0.18)
        x_right = left.px(idx + 0.18)
        y_top = left.py(record.total_time_hours)
        svg.rect(x_left, y_top, x_right - x_left, left.py(left.ymin) - y_top, fill=MODE_COLORS[record.mode], stroke=AXIS, stroke_width=0.7)
        svg.text((x_left + x_right) / 2, y_top - 8, f"{record.total_time_hours:.2f} h", size=10, family=SANS)
        x = right.px(record.total_time_hours)
        y = right.py(record.final_test_acc)
        svg.circle(x, y, 7.0, fill=MODE_COLORS[record.mode], stroke=AXIS, stroke_width=0.8)
        svg.text(x + 10, y - 8, MODE_LABELS[record.mode], size=11, anchor="start", family=SANS)
    runtime_ratio = fair[0].total_time_hours / fair[1].total_time_hours if fair and fair[1].total_time_hours else 0.0
    svg.text(735, 370, f"Intrinsic Dim is {runtime_ratio:.2f}x slower than LoRA on seed 43.", size=12, family=SANS)
    svg.save(output_dir / "runtime_tradeoff_seed43.svg")


def series_to_points(plot: PlotArea, values: Sequence[float], x_start: int = 1) -> List[tuple[float, float]]:
    return [(plot.px(idx + x_start), plot.py(value)) for idx, value in enumerate(values)]


def draw_small_legend(svg: Svg, x: float, y: float) -> None:
    legend_box(svg, x, y, [
        ("dash", SEED_COLORS[42], "seed 42 train"),
        ("line", SEED_COLORS[42], "seed 42 test"),
        ("dash", SEED_COLORS[43], "seed 43 train"),
        ("line", SEED_COLORS[43], "seed 43 test"),
    ])


def save_learning_curves(records: List[RunRecord], output_dir: Path) -> None:
    svg = Svg(1180, 900, "Learning Curves")
    id_records = sorted([r for r in records if r.mode == "id_module"], key=lambda r: r.seed)
    lora_records = sorted([r for r in records if r.mode == "lora"], key=lambda r: r.seed)
    acc_values = []
    loss_values = []
    for record in records:
        acc_values.extend(record.train_acc_history)
        acc_values.extend(record.test_acc_history)
        loss_values.extend(record.train_loss_history)
        loss_values.extend(record.test_loss_history)
    acc_min, acc_max = pad_range(acc_values, 0.05)
    acc_min = min(acc_min, 86.0)
    acc_max = max(acc_max, 100.5)
    loss_min, loss_max = pad_range(loss_values, 0.08)
    plots = {
        "id_acc": PlotArea(90, 90, 440, 260, 1, 20, acc_min, acc_max),
        "id_loss": PlotArea(650, 90, 440, 260, 1, 20, loss_min, loss_max),
        "lora_acc": PlotArea(90, 500, 440, 260, 1, 20, acc_min, acc_max),
        "lora_loss": PlotArea(650, 500, 440, 260, 1, 20, loss_min, loss_max),
    }
    draw_plot_frame(svg, plots["id_acc"], "Intrinsic Dim Accuracy Curves", "Epoch", "Accuracy (%)", [1, 5, 10, 15, 20], nice_ticks(acc_min, acc_max, 6), lambda t: f"{int(t)}", lambda t: f"{t:.0f}")
    draw_plot_frame(svg, plots["id_loss"], "Intrinsic Dim Loss Curves", "Epoch", "Loss", [1, 5, 10, 15, 20], nice_ticks(loss_min, loss_max, 6), lambda t: f"{int(t)}", lambda t: f"{t:.2f}")
    draw_plot_frame(svg, plots["lora_acc"], "LoRA Accuracy Curves", "Epoch", "Accuracy (%)", [1, 5, 10, 15, 20], nice_ticks(acc_min, acc_max, 6), lambda t: f"{int(t)}", lambda t: f"{t:.0f}")
    draw_plot_frame(svg, plots["lora_loss"], "LoRA Loss Curves", "Epoch", "Loss", [1, 5, 10, 15, 20], nice_ticks(loss_min, loss_max, 6), lambda t: f"{int(t)}", lambda t: f"{t:.2f}")
    for record in id_records:
        color = SEED_COLORS.get(record.seed, "#374151")
        svg.polyline(series_to_points(plots["id_acc"], record.train_acc_history), stroke=color, stroke_width=2.0, dash="7 5", opacity=0.88)
        svg.polyline(series_to_points(plots["id_acc"], record.test_acc_history), stroke=color, stroke_width=2.3, opacity=0.98)
        svg.polyline(series_to_points(plots["id_loss"], record.train_loss_history), stroke=color, stroke_width=2.0, dash="7 5", opacity=0.88)
        svg.polyline(series_to_points(plots["id_loss"], record.test_loss_history), stroke=color, stroke_width=2.3, opacity=0.98)
    for record in lora_records:
        color = SEED_COLORS.get(record.seed, "#374151")
        svg.polyline(series_to_points(plots["lora_acc"], record.train_acc_history), stroke=color, stroke_width=2.0, dash="7 5", opacity=0.88)
        svg.polyline(series_to_points(plots["lora_acc"], record.test_acc_history), stroke=color, stroke_width=2.3, opacity=0.98)
        svg.polyline(series_to_points(plots["lora_loss"], record.train_loss_history), stroke=color, stroke_width=2.0, dash="7 5", opacity=0.88)
        svg.polyline(series_to_points(plots["lora_loss"], record.test_loss_history), stroke=color, stroke_width=2.3, opacity=0.98)
    draw_small_legend(svg, 930, 120)
    svg.save(output_dir / "learning_curves.svg")


def save_id_tuning_trajectory(tuning_summary: Path, output_dir: Path) -> None:
    payload = load_json(tuning_summary)
    stage_entries = payload["modes"]["id_module"]["stages"]
    labels: List[str] = []
    values: List[float] = []
    stage_ids: List[int] = []
    for stage in stage_entries:
        stage_id = int(stage["stage"])
        for result in stage["results"]:
            labels.append(f"s{stage_id}-{result['config_id']}")
            values.append(float(result["best_test_acc"]))
            stage_ids.append(stage_id)
    svg = Svg(1120, 520, "Intrinsic-Dim Tuning Trajectory")
    plot = PlotArea(90, 80, 900, 290, -0.5, len(labels) - 0.5, 75.0, 93.0)
    draw_plot_frame(svg, plot, "Intrinsic-Dim Tuning Trajectory", "Candidate / stage", "Best validation accuracy (%)", list(range(len(labels))), [75, 80, 85, 90], lambda t: labels[int(round(t))], lambda t: f"{t:.0f}", x_tick_rotation=40)
    for idx, value in enumerate(values):
        x_left = plot.px(idx - 0.34)
        x_right = plot.px(idx + 0.34)
        y_top = plot.py(value)
        svg.rect(x_left, y_top, x_right - x_left, plot.py(plot.ymin) - y_top, fill=STAGE_COLORS[stage_ids[idx]], stroke=AXIS, stroke_width=0.6)
        svg.text((x_left + x_right) / 2, y_top - 8, f"{value:.2f}", size=9, family=SANS)
    svg.line(plot.x, plot.py(91.82), plot.x + plot.width, plot.py(91.82), stroke="#0f766e", stroke_width=1.5, dash="8 5")
    svg.line(plot.x, plot.py(92.06), plot.x + plot.width, plot.py(92.06), stroke="#134e4a", stroke_width=1.5, dash="2 6")
    legend_box(svg, 830, 92, [("bar", STAGE_COLORS[1], "Stage 1 (5 epochs)"), ("bar", STAGE_COLORS[2], "Stage 2 (10 epochs)"), ("bar", STAGE_COLORS[3], "Stage 3 (20 epochs)"), ("dash", "#0f766e", "Final seed 42"), ("dash", "#134e4a", "Final seed 43")])
    svg.save(output_dir / "id_tuning_trajectory.svg")


def save_id_dimension_sensitivity(records: List[RunRecord], output_dir: Path) -> None:
    svg = Svg(840, 460, "Intrinsic-Dim Sensitivity to Subspace Dimension")
    labels = []
    values = []
    colors = []
    for record in records:
        dim = int(record.extra.get("subspace_dim") or -1)
        labels.append(f"d={dim}\nseed {record.seed}")
        values.append(record.best_test_acc)
        colors.append("#991b1b" if record.best_test_acc < 80 else MODE_COLORS["id_module"])
    plot = PlotArea(90, 80, 650, 280, -0.5, len(labels) - 0.5, 45.0, 95.0)
    draw_plot_frame(svg, plot, "Intrinsic-Dim Sensitivity to Subspace Dimension", "Configuration", "Best test accuracy (%)", list(range(len(labels))), [50, 60, 70, 80, 90], lambda t: labels[int(round(t))], lambda t: f"{t:.0f}")
    for idx, value in enumerate(values):
        x_left = plot.px(idx - 0.28)
        x_right = plot.px(idx + 0.28)
        y_top = plot.py(value)
        svg.rect(x_left, y_top, x_right - x_left, plot.py(plot.ymin) - y_top, fill=colors[idx], stroke=AXIS, stroke_width=0.7)
        svg.text((x_left + x_right) / 2, y_top - 8, f"{value:.2f}", size=10, family=SANS)
    svg.text(760, 160, "The early", size=11, anchor="start", family=SANS)
    svg.text(760, 178, "d=371812 run", size=11, anchor="start", family=SANS)
    svg.text(760, 196, "collapsed.", size=11, anchor="start", family=SANS)
    svg.text(760, 236, "d=666724 plus", size=11, anchor="start", family=SANS)
    svg.text(760, 254, "longer training", size=11, anchor="start", family=SANS)
    svg.text(760, 272, "recovered >91.8%.", size=11, anchor="start", family=SANS)
    svg.save(output_dir / "id_dimension_sensitivity.svg")


def write_summary_markdown(records: List[RunRecord], summary: Dict[str, Dict[str, float]], output_dir: Path) -> None:
    id_mean = summary["id_module"]["final_mean"]
    lora_mean = summary["lora"]["final_mean"]
    id_best = summary["id_module"]["best_mean"]
    lora_best = summary["lora"]["best_mean"]
    delta_final = lora_mean - id_mean
    delta_best = lora_best - id_best
    fair_id = next(r for r in records if r.mode == "id_module" and r.seed == 43)
    fair_lora = next(r for r in records if r.mode == "lora" and r.seed == 43)
    ratio = fair_id.total_time_hours / fair_lora.total_time_hours
    lines = [
        "# Generated Final Analysis Summary",
        "",
        f"- Intrinsic-dim mean final accuracy: {id_mean:.2f}%",
        f"- LoRA mean final accuracy: {lora_mean:.2f}%",
        f"- Final accuracy delta (LoRA - ID): {delta_final:.2f} points",
        f"- Best accuracy delta (LoRA - ID): {delta_best:.2f} points",
        f"- Fair runtime comparison on seed 43: Intrinsic Dim {fair_id.total_time_hours:.2f} h vs LoRA {fair_lora.total_time_hours:.2f} h ({ratio:.2f}x slower)",
        "",
        "Use seed 43 for runtime comparison because the seed 42 intrinsic-dim run was resumed across a local machine and a cloud host.",
    ]
    (output_dir / "final_analysis.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    ensure_dir(args.output_dir)
    all_records = load_records(args.results_root)
    seeds = [int(token.strip()) for token in args.seeds.split(",") if token.strip()]
    final_records = select_final_records(all_records, args.dataset, args.model_name, seeds, args.id_subspace_dim, args.lora_rank)
    if len(final_records) != 4:
        raise SystemExit(f"Expected 4 final records, found {len(final_records)}. Check dataset/model/seeds filters.")
    summary = summarize(final_records)
    write_csvs(final_records, summary, args.output_dir)
    write_summary_markdown(final_records, summary, args.output_dir)
    save_accuracy_by_seed(final_records, args.output_dir)
    save_mean_accuracy_and_gap(summary, args.output_dir)
    save_runtime_tradeoff(final_records, args.output_dir)
    save_learning_curves(final_records, args.output_dir)
    save_id_tuning_trajectory(args.tuning_summary, args.output_dir)
    save_id_dimension_sensitivity(select_dimension_records(all_records, args.dataset, args.model_name, args.id_subspace_dim), args.output_dir)
    print(f"Generated figures in {args.output_dir}")


if __name__ == "__main__":
    main()
