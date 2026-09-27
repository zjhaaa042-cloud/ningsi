"""结构化状态评估报告（文档 7.2.10）。

报告包含状态评分、问题分析、改善建议，并逐条回填证据：每条结论都能指回所用指标、
窗口统计与参数版本，使结果可复算、可追溯。
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

from ningsi import config
from ningsi.assessment.joint import JointAssessment


@dataclass
class SessionReport:
    participant: str
    session: str
    run: str
    indicators: dict = field(default_factory=dict)
    quality: dict = field(default_factory=dict)
    scales: dict = field(default_factory=dict)
    behavior: dict = field(default_factory=dict)
    assessment: JointAssessment = None
    extras: dict = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "spec": config.LABEL_SPEC,
            "participant": self.participant,
            "session": self.session,
            "run": self.run,
            "versions": {
                "product": config.VERSION,
                "spectrum": config.SPECTRUM_SPEC,
                "indicator": config.INDICATOR_SPEC,
                "baseline": config.BASELINE_SPEC,
                "assessment": config.LABEL_SPEC,
            },
            "indicators": self.indicators,
            "quality": self.quality,
            "scales": self.scales,
            "behavior": self.behavior,
            "assessment": self.assessment.as_dict() if self.assessment else None,
            "extras": self.extras,
        }

    def to_json(self, indent: int = 2) -> str:
        return json.dumps(self.as_dict(), ensure_ascii=False, indent=indent)

    def to_markdown(self) -> str:
        a = self.assessment
        lines = [
            f"# 凝思状态评估报告（sub-{self.participant} / ses-{self.session} / run-{self.run}）",
            "",
            f"- 指标口径：{config.INDICATOR_SPEC}；基线口径：{config.BASELINE_SPEC}；处理链：{config.SPECTRUM_SPEC}",
            f"- 软件版本：{config.VERSION}",
            "",
            "## 一、状态评分",
        ]
        indicator_summary = self.indicators.get("summary", {}) if isinstance(self.indicators, dict) else {}
        labels = {"focus": "专注度", "relax": "放松度", "load": "认知负荷"}
        if indicator_summary:
            for key, label in labels.items():
                stat = indicator_summary.get(key)
                if stat:
                    lines.append(f"- {label}：{stat['mean']:.2f}（波动 {stat['std']:.2f}，n={stat['n']}）")
        else:
            lines.append("- 无可用指标（见采集质量）")
        if isinstance(self.indicators, dict) and self.indicators.get("band_z"):
            band_text = "、".join(f"{k} {v:+.2f}" for k, v in sorted(self.indicators["band_z"].items()))
            lines.append(f"- 相对基线的频带 z 值：{band_text}")

        lines += ["", "## 二、采集质量"]
        q = self.quality or {}
        lines.append(
            f"- 4 秒窗 {q.get('windows_usable', 0)}/{q.get('windows_total', 0)} 可用"
            f"（可用窗比例 {q.get('valid_ratio', 0):.0%}）"
        )
        if q.get("unusable_reasons"):
            detail = "、".join(f"{k}×{v}" for k, v in sorted(q["unusable_reasons"].items()))
            lines.append(f"- 排除原因：{detail}（含伪迹窗不计入指标）")

        lines += ["", "## 三、量表结果"]
        if self.scales:
            for code, result in sorted(self.scales.items()):
                lines.append(
                    f"- {result['name']}（{code}）：粗分 {result['raw_score']}，标准分 {result['standard_score']}，"
                    f"{result['level']}；反向题 {result['reverse_items']}，版本 {result['version']}"
                )
        else:
            lines.append("- 未采集")

        lines += ["", "## 四、行为任务"]
        if self.behavior:
            for name, result in sorted(self.behavior.items()):
                if name == "sart":
                    lines.append(
                        f"- SART：{result.get('trials', '—')} 试次（No-Go {result.get('nogo_trials', '—')}），"
                        f"正确率 {result.get('go_accuracy', 0):.0%}，虚报率 {result.get('commission_rate', 0):.1%}，"
                        f"漏报率 {result.get('omission_rate', 0):.1%}，反应时 {result.get('rt_mean', 0):.3f}"
                        f"±{result.get('rt_sd', 0):.3f}s，序列集 {result.get('sequence_set_id', '—')}"
                    )
                elif name == "pvt":
                    lines.append(
                        f"- PVT-B：{result.get('trials', '—')} 试次，中位反应时 {result.get('rt_median', 0):.3f}s，"
                        f"慢反应率 {result.get('lapse_rate', 0):.1%}，抢答 {result.get('false_starts', 0)} 次"
                    )
                else:
                    lines.append(f"- {name}：{json.dumps(result, ensure_ascii=False)}")
        else:
            lines.append("- 未采集")

        lines += ["", "## 五、联合评估结论", f"- 结论：{a.conclusion if a else '未评估'}"]
        if a:
            for dimension, state in sorted(a.dimension_states.items()):
                label = {"attention": "专注维度", "stress": "压力维度"}.get(dimension, dimension)
                lines.append(f"- {label}：{state['state']}（依据 {', '.join(state['evidence_codes']) or '无'}）")
            lines.append(f"- 一致性：脑电 vs 量表 {a.consistency['eeg_vs_scale']}；脑电 vs 行为 {a.consistency['eeg_vs_behavior']}")
            lines += ["", "## 六、改善建议"]
            for item in a.advice:
                lines.append(f"- {item}")
            lines += ["", "## 七、证据回填"]
            for item in a.evidence:
                state = "可用" if item.available else "不可用"
                lines.append(f"- [{item.code}]（{item.source}/{state}）{item.summary}；引用 {json.dumps(item.ref, ensure_ascii=False)}")
            lines += ["", f"> 边界说明：{a.boundary}"]
        return "\n".join(lines)
