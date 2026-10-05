"""Warmup-stable-decay (WSD / "trapezoid") learning-rate schedule.

Linear warmup, a long flat stretch at full LR, then a linear cooldown to zero.
Recent open models (MiniCPM, DeepSeek, the Muon speedruns) prefer this over
cosine because the flat middle means a run can be extended — or checkpoints
branched into a cooldown — without re-fitting the whole schedule to a new
horizon.

Returns a multiplier in [0, 1]; the caller applies it to each optimizer
group's base LR every step.
"""


def wsd_multiplier(
    step: int, total_steps: int, warmup_steps: int, cooldown_steps: int
) -> float:
    if step < warmup_steps:
        return (step + 1) / warmup_steps
    if step < total_steps - cooldown_steps:
        return 1.0
    return (total_steps - step) / cooldown_steps
