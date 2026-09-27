"""闭环神经反馈训练（文档 7.2.8、7.2.9、8.6）。

反馈以专注度评分驱动；训练分段进行，每段记录平均专注度、达标时间占比与波动幅度；
下一段目标按上一段表现自适应调整；训练前后各采集一次静息基线用于对比。
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ningsi import config
from ningsi.signal.baseline import Baseline, can_compare
from ningsi.signal.indicators import score_from_z


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def initial_target(baseline: Baseline | None = None) -> tuple[float, str]:
    """初始目标取自基线分布中位数：中位数对应 z=0、评分 0.5，再做区间裁剪。"""
    settings = config.TRAINING
    base_score = score_from_z(0.0)
    target = clamp(base_score, settings["target_min"], settings["target_max"])
    rationale = f"基线中位数对应评分 {base_score:.2f}，裁剪到 [{settings['target_min']:.2f}, {settings['target_max']:.2f}] 后取 {target:.2f}"
    if baseline is not None and not baseline.valid:
        rationale += "；注意本次基线质量不合格，目标仅作演示用途"
    return target, rationale


@dataclass
class Segment:
    index: int
    duration_sec: float
    target: float
    hold_sec: float
    samples: list = field(default_factory=list)   # [(t, focus_score)]
    excluded_windows: int = 0

    def stats(self) -> dict:
        values = [value for _, value in self.samples]
        if not values:
            return {"mean": 0.0, "on_target_ratio": 0.0, "volatility": 0.0, "n": 0, "achieved": False}
        mean = sum(values) / len(values)
        if len(values) > 1:
            volatility = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        else:
            volatility = 0.0
        on_target = sum(1 for v in values if v >= self.target)
        return {
            "mean": round(mean, 4),
            "on_target_ratio": round(on_target / len(values), 4),
            "volatility": round(volatility, 4),
            "n": len(values),
            "achieved": (on_target / len(values)) >= 0.5,
        }

    def as_dict(self) -> dict:
        return {
            "index": self.index,
            "duration_sec": self.duration_sec,
            "target": round(self.target, 4),
            "hold_sec": self.hold_sec,
            "excluded_windows": self.excluded_windows,
            "stats": self.stats(),
        }


@dataclass
class NeurofeedbackSession:
    participant: str
    session: str
    run: str
    mode: str = "quick"
    target: float = 0.5
    hold_sec: float = config.TRAINING["hold_sec"]
    segments: list = field(default_factory=list)
    rationale: str = ""
    baseline_before: dict = field(default_factory=dict)
    baseline_after: dict = field(default_factory=dict)
    spec: str = "neurofeedback-v1"

    def plan(self) -> dict:
        return dict(config.TRAINING[self.mode])

    def next_target(self, stats: dict) -> float:
        """自适应：达标时间占比高且波动小 → 提高要求；占比低或波动大 → 降低要求。"""
        settings = config.TRAINING
        step = settings["target_step"]
        target, hold = self.target, self.hold_sec
        if stats.get("n", 0) == 0:
            return target
        if stats["on_target_ratio"] >= settings["on_target_raise"] and stats["volatility"] <= settings["volatility_high"]:
            target = clamp(target + step, settings["target_min"], settings["target_max"])
            self.hold_sec = min(self.hold_sec + 2.0, 20.0)
        elif stats["on_target_ratio"] < settings["on_target_lower"] or stats["volatility"] > settings["volatility_high"]:
            target = clamp(target - step, settings["target_min"], settings["target_max"])
            self.hold_sec = max(self.hold_sec - 2.0, 3.0)
        return target

    def add_segment(self, samples, duration_sec: float, excluded_windows: int = 0) -> Segment:
        segment = Segment(
            index=len(self.segments) + 1,
            duration_sec=float(duration_sec),
            target=self.target,
            hold_sec=self.hold_sec,
            samples=[(float(t), float(v)) for t, v in samples],
            excluded_windows=int(excluded_windows),
        )
        self.segments.append(segment)
        self.target = self.next_target(segment.stats())
        return segment

    def summary(self, baseline_before: Baseline | None = None, baseline_after: Baseline | None = None) -> dict:
        stats = [segment.stats() for segment in self.segments]
        means = [s["mean"] for s in stats]
        change = round(means[-1] - means[0], 4) if len(means) > 1 else 0.0
        comparable, reason = (True, "no_baseline")
        if baseline_before is not None and baseline_after is not None:
            comparable, reason = can_compare(baseline_before, baseline_after.srate, baseline_after.channels, baseline_after.device)
        return {
            "spec": self.spec,
            "mode": self.mode,
            "segments": [segment.as_dict() for segment in self.segments],
            "target_plan": self.plan(),
            "target_rationale": self.rationale,
            "initial_target": round(self.target, 4) if not self.segments else self.segments[0].target,
            "final_target": round(self.target, 4),
            "mean_focus": round(sum(means) / len(means), 4) if means else 0.0,
            "on_target_ratio": round(sum(s["on_target_ratio"] for s in stats) / len(stats), 4) if stats else 0.0,
            "volatility": round(sum(s["volatility"] for s in stats) / len(stats), 4) if stats else 0.0,
            "first_to_last_change": change,
            "baseline_comparable": comparable,
            "baseline_comparability": reason,
            "baseline_before": baseline_before.as_dict() if baseline_before else None,
            "baseline_after": baseline_after.as_dict() if baseline_after else None,
        }
