#!/usr/bin/env python3
"""计算加权决策矩阵，并执行逐项权重敏感性测试。"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any


EPSILON = 1e-9


class InputError(ValueError):
    pass


class ChineseArgumentParser(argparse.ArgumentParser):
    def format_help(self) -> str:
        return (
            super()
            .format_help()
            .replace("usage:", "用法：", 1)
            .replace("positional arguments:", "位置参数：")
            .replace("optional arguments:", "可选参数：")
            .replace("options:", "选项：")
        )


def require_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise InputError(f"{label} 必须是数字")
    number = float(value)
    if not math.isfinite(number):
        raise InputError(f"{label} 必须是有限数值")
    return number


def load_matrix(path: Path) -> tuple[list[str], list[float], list[dict[str, Any]]]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError as exc:
        raise InputError(f"找不到输入文件：{path}") from exc
    except json.JSONDecodeError as exc:
        raise InputError(f"JSON 格式无效：{exc}") from exc

    if not isinstance(payload, dict):
        raise InputError("JSON 顶层值必须是对象")

    raw_criteria = payload.get("criteria")
    raw_options = payload.get("options")
    if not isinstance(raw_criteria, list) or not raw_criteria:
        raise InputError("criteria 必须是非空列表")
    if not isinstance(raw_options, list) or len(raw_options) < 2:
        raise InputError("options 必须至少包含两个方案")

    criterion_names: list[str] = []
    raw_weights: list[float] = []
    for index, item in enumerate(raw_criteria):
        if not isinstance(item, dict):
            raise InputError(f"criteria[{index}] 必须是对象")
        name = item.get("name")
        if not isinstance(name, str) or not name.strip():
            raise InputError(f"criteria[{index}].name 必须是非空字符串")
        name = name.strip()
        if name in criterion_names:
            raise InputError(f"指标名称重复：{name}")
        weight = require_number(item.get("weight"), f"criteria[{index}].weight")
        if weight < 0:
            raise InputError(f"criteria[{index}].weight 不能为负数")
        criterion_names.append(name)
        raw_weights.append(weight)

    weight_sum = sum(raw_weights)
    if weight_sum <= EPSILON:
        raise InputError("至少一个指标的权重必须大于零")
    weights = [weight / weight_sum for weight in raw_weights]

    options: list[dict[str, Any]] = []
    option_names: set[str] = set()
    for index, item in enumerate(raw_options):
        if not isinstance(item, dict):
            raise InputError(f"options[{index}] 必须是对象")
        name = item.get("name")
        scores = item.get("scores")
        if not isinstance(name, str) or not name.strip():
            raise InputError(f"options[{index}].name 必须是非空字符串")
        name = name.strip()
        if name in option_names:
            raise InputError(f"方案名称重复：{name}")
        if not isinstance(scores, dict):
            raise InputError(f"options[{index}].scores 必须是对象")

        normalized_scores: dict[str, float] = {}
        for criterion in criterion_names:
            if criterion not in scores:
                raise InputError(f"方案 {name!r} 缺少指标 {criterion!r} 的评分")
            score = require_number(scores[criterion], f"评分 {name}.{criterion}")
            if score < 0 or score > 10:
                raise InputError(f"评分 {name}.{criterion} 必须在 0 到 10 之间")
            normalized_scores[criterion] = score

        option_names.add(name)
        options.append({"name": name, "scores": normalized_scores})

    return criterion_names, weights, options


def totals_for(
    criterion_names: list[str], weights: list[float], options: list[dict[str, Any]]
) -> dict[str, float]:
    return {
        option["name"]: sum(
            option["scores"][criterion] * weights[index]
            for index, criterion in enumerate(criterion_names)
        )
        for option in options
    }


def leaders(totals: dict[str, float]) -> list[str]:
    best = max(totals.values())
    return sorted(name for name, score in totals.items() if abs(score - best) <= EPSILON)


def adjust_weight(weights: list[float], index: int, delta: float) -> list[float]:
    if len(weights) == 1:
        return weights[:]

    target = min(1.0, max(0.0, weights[index] + delta))
    remainder = 1.0 - target
    other_sum = sum(weight for i, weight in enumerate(weights) if i != index)
    adjusted: list[float] = []
    for i, weight in enumerate(weights):
        if i == index:
            adjusted.append(target)
        elif other_sum <= EPSILON:
            adjusted.append(remainder / (len(weights) - 1))
        else:
            adjusted.append(weight / other_sum * remainder)
    return adjusted


def analyze(
    criterion_names: list[str],
    weights: list[float],
    options: list[dict[str, Any]],
    sensitivity_step: float,
) -> dict[str, Any]:
    base_totals = totals_for(criterion_names, weights, options)
    base_leaders = leaders(base_totals)
    rankings = sorted(
        ({"name": name, "score": round(score, 6)} for name, score in base_totals.items()),
        key=lambda item: (-item["score"], item["name"]),
    )

    scenarios: list[dict[str, Any]] = []
    for index, criterion in enumerate(criterion_names):
        for direction, delta in (("down", -sensitivity_step), ("up", sensitivity_step)):
            adjusted = adjust_weight(weights, index, delta)
            if all(abs(a - b) <= EPSILON for a, b in zip(adjusted, weights)):
                continue
            scenario_totals = totals_for(criterion_names, adjusted, options)
            scenario_leaders = leaders(scenario_totals)
            scenarios.append(
                {
                    "criterion": criterion,
                    "direction": direction,
                    "weight": round(adjusted[index], 8),
                    "leaders": scenario_leaders,
                    "changed": scenario_leaders != base_leaders,
                }
            )

    return {
        "criteria": [
            {"name": name, "weight": round(weights[index], 8)}
            for index, name in enumerate(criterion_names)
        ],
        "rankings": rankings,
        "base_leaders": base_leaders,
        "sensitivity": {
            "step": sensitivity_step,
            "stable": not any(scenario["changed"] for scenario in scenarios),
            "scenarios": scenarios,
        },
    }


def markdown_report(
    result: dict[str, Any], criterion_names: list[str], options: list[dict[str, Any]]
) -> str:
    score_lookup = {item["name"]: item["score"] for item in result["rankings"]}
    rank_lookup = {item["name"]: index + 1 for index, item in enumerate(result["rankings"])}
    weight_lookup = {item["name"]: item["weight"] for item in result["criteria"]}

    headers = ["排名", "方案"] + [
        f"{name} ({weight_lookup[name]:.0%})" for name in criterion_names
    ] + ["加权总分"]
    rows = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for option in sorted(options, key=lambda item: rank_lookup[item["name"]]):
        values = [str(rank_lookup[option["name"]]), option["name"]]
        values.extend(f"{option['scores'][name]:.2f}" for name in criterion_names)
        values.append(f"{score_lookup[option['name']]:.3f}")
        rows.append("| " + " | ".join(values) + " |")

    sensitivity = result["sensitivity"]
    changed = [scenario for scenario in sensitivity["scenarios"] if scenario["changed"]]
    lines = [
        "# 加权决策矩阵",
        "",
        *rows,
        "",
        f"**基础领先方案：** {', '.join(result['base_leaders'])}",
        f"**权重敏感性：** 各指标权重单独变化 ±{sensitivity['step']:.0%} 时，结果{'稳健' if sensitivity['stable'] else '脆弱'}。",
    ]
    if changed:
        lines.extend(["", "## 会改变排名的情景"])
        for scenario in changed:
            direction = "下调" if scenario["direction"] == "down" else "上调"
            lines.append(
                f"- “{scenario['criterion']}”权重{direction}至 "
                f"{scenario['weight']:.1%}：领先方案变为 {', '.join(scenario['leaders'])}。"
            )
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = ChineseArgumentParser(
        description="使用 0～10 分制计算加权决策矩阵，并测试权重敏感性。",
        add_help=False,
    )
    parser.add_argument("-h", "--help", action="help", help="显示帮助信息并退出")
    parser.add_argument("input", type=Path, help="决策矩阵 JSON 文件路径")
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="输出格式：Markdown 或 JSON（默认：markdown）",
    )
    parser.add_argument(
        "--sensitivity-step",
        type=float,
        default=0.10,
        help="逐项敏感性测试采用的绝对归一化权重变化量（默认：0.10）",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if not 0 < args.sensitivity_step < 1:
        print("错误：--sensitivity-step 必须在 0 到 1 之间", file=sys.stderr)
        return 2
    try:
        criterion_names, weights, options = load_matrix(args.input)
        result = analyze(criterion_names, weights, options, args.sensitivity_step)
    except InputError as exc:
        print(f"错误：{exc}", file=sys.stderr)
        return 2

    if args.format == "json":
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        print(markdown_report(result, criterion_names, options), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
