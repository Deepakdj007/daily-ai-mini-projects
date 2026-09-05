"""The ablation ladder: five configurations, one new mechanism per rung.

Inputs:  none
Outputs: CONFIGS, the list the evaluation harness iterates over

Keeping the experiment's shape in its own file makes one thing obvious at a
glance - every row of the leaderboard differs from the row above it by
exactly one switch.
"""

from __future__ import annotations

from src.guard import GuardConfig

# The ladder. Each rung adds one mechanism to the one above it.
CONFIGS: list[GuardConfig] = [
    GuardConfig("off"),
    GuardConfig("prompt-only", hardened_prompt=True),
    GuardConfig(
        "classifier",
        hardened_prompt=True,
        normalize_text=True,
        scan_user=True,
        scan_docs=True,
    ),
    GuardConfig(
        "classifier+judge",
        hardened_prompt=True,
        normalize_text=True,
        scan_user=True,
        escalate_to_judge=True,
        scan_docs=True,
    ),
    GuardConfig(
        "full stack",
        hardened_prompt=True,
        normalize_text=True,
        scan_user=True,
        escalate_to_judge=True,
        scan_docs=True,
        pii_firewall=True,
        canary_check=True,
    ),
]

CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}



CONFIGS_BY_NAME = {c.name: c for c in CONFIGS}
