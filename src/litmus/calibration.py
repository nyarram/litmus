"""Judge calibration: measure an LLM judge against human labels before trusting it.

A judge is a model with opinions. Calibration answers the only question
that matters: how often does the judge agree with a human on pass/fail?
Ship the judge only if agreement clears your bar (0.8+ is a sane default).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Disagreement:
    case_id: str
    judge_passed: bool | None
    human_passed: bool
    judge_explanation: str | None = None


@dataclass
class CalibrationReport:
    n: int
    agreement_rate: float
    judge_pass_rate: float
    human_pass_rate: float
    disagreements: list[Disagreement] = field(default_factory=list)

    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "agreement_rate": round(self.agreement_rate, 4),
            "judge_pass_rate": round(self.judge_pass_rate, 4),
            "human_pass_rate": round(self.human_pass_rate, 4),
            "n_disagreements": len(self.disagreements),
            "disagreements": [d.__dict__ for d in self.disagreements],
        }


def calibrate(
    judge_verdicts: dict[str, tuple[bool | None, str | None]],
    human_labels: dict[str, bool],
) -> CalibrationReport:
    """Compare judge pass/fail against human pass/fail.

    Args:
        judge_verdicts: mapping case_id -> (judge passed, judge explanation)
        human_labels: mapping case_id -> human passed
    """
    shared = [cid for cid in judge_verdicts if cid in human_labels]
    if not shared:
        raise ValueError("No overlapping case ids between judge verdicts and human labels")
    disagreements: list[Disagreement] = []
    agree = 0
    for cid in shared:
        j_passed, j_expl = judge_verdicts[cid]
        h_passed = human_labels[cid]
        if j_passed == h_passed:
            agree += 1
        else:
            disagreements.append(
                Disagreement(
                    case_id=cid,
                    judge_passed=j_passed,
                    human_passed=h_passed,
                    judge_explanation=j_expl,
                )
            )
    n = len(shared)
    return CalibrationReport(
        n=n,
        agreement_rate=agree / n,
        judge_pass_rate=sum(1 for c in shared if judge_verdicts[c][0]) / n,
        human_pass_rate=sum(1 for c in shared if human_labels[c]) / n,
        disagreements=disagreements,
    )
