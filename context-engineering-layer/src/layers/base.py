"""The contract every layer signs: one config, one dependency bundle, one call.

Inputs:  nothing from config.py - defaults are literals here
Outputs: PipelineConfig, LayerDeps, the Layer protocol

Defaults live here as plain numbers rather than being read from src.config,
because that is what lets src/layers/ be copied into another codebase without
the experiment coming with it. The experiment overrides them in ladder.py.

PipelineConfig is frozen so an ablation arm is provably a one-field change:
`replace(full, retrieve=False)` cannot accidentally move a second knob, and a
typo raises instead of silently doing nothing.
"""

from __future__ import annotations

from dataclasses import dataclass, fields, replace
from typing import ClassVar, Protocol

from src.context import Context
from src.types import SummarizeFn, TokenCounter


@dataclass(frozen=True, slots=True)
class PipelineConfig:
    """Which layers run, and the one or two knobs each of them owns."""

    # --- switches: an ablation arm flips exactly one of these ---
    history: bool = True
    cap: bool = True
    pin: bool = True
    retrieve: bool = True
    window: bool = True
    summarize: bool = True

    # --- shared ---
    context_budget: int = 900

    # --- cap: a pure function of the message, which is what makes it
    #     prefix-stable. No budget, no position, no conversation length. ---
    cap_max_message_tokens: int = 35
    cap_head_tokens: int = 22
    cap_tail_tokens: int = 8

    # --- pin ---
    pin_max_tokens: int = 40

    # --- retrieve: a FIXED reservation, taken whether or not de-duplication
    #     drops a hit. Handing freed tokens back to the window would change
    #     how many turns it keeps per query, which moves a block in front of
    #     the prefix divergence point and forfeits the cache. ---
    retrieve_k: int = 2
    retrieve_block_tokens: int = 280
    retrieve_min_score: float = 0.0

    # --- window ---
    window_min_turns: int = 1

    # --- summarize ---
    summary_budget_tokens: int = 80
    summary_trigger_ratio: float = 1.0

    SWITCHES: ClassVar[tuple[str, ...]] = (
        "history", "cap", "pin", "retrieve", "window", "summarize",
    )

    def enabled(self, layer: str) -> bool:
        """True if `layer` should run in this configuration."""
        return bool(getattr(self, layer))

    def switch_map(self) -> dict[str, bool]:
        """Just the on/off switches, for the manifest and the report."""
        return {name: bool(getattr(self, name)) for name in self.SWITCHES}

    def without(self, layer: str) -> PipelineConfig:
        """A copy with exactly one layer turned off."""
        return replace(self, **{layer: False})

    def with_(self, layer: str, **knobs: object) -> PipelineConfig:
        """A copy with one layer turned on, plus any knobs it needs."""
        return replace(self, **{layer: True}, **knobs)

    def diff(self, other: PipelineConfig) -> tuple[str, ...]:
        """Every field that differs. Compares all fields, not just switches.

        Budget-only variants differ in no switch at all, so a switch-only
        comparison would report them as identical to their parent and the
        single-change invariant would pass without meaning anything.
        """
        return tuple(
            f.name
            for f in fields(self)
            if getattr(self, f.name) != getattr(other, f.name)
        )


@dataclass(frozen=True, slots=True)
class LayerDeps:
    """Capabilities a layer may use, injected rather than imported.

    This is why src/layers/ contains no network code: `summarize` receives a
    function, so the whole pipeline assembles offline and every layer can be
    smoke-tested without an API key.
    """

    count: TokenCounter
    summarize: SummarizeFn | None = None


class Layer(Protocol):
    """The single shape every src/layers/*.py exposes as module-level `apply`."""

    async def __call__(
        self, ctx: Context, cfg: PipelineConfig, deps: LayerDeps
    ) -> Context: ...


if __name__ == "__main__":
    full = PipelineConfig()
    assert full.enabled("retrieve")
    no_ret = full.without("retrieve")
    assert full.diff(no_ret) == ("retrieve",), full.diff(no_ret)
    assert not no_ret.enabled("retrieve")

    smaller = replace(full, context_budget=400)
    assert full.diff(smaller) == ("context_budget",), (
        "a budget-only variant must still register as a one-field change"
    )
    assert full.switch_map() == smaller.switch_map(), (
        "...even though no switch moved - which is why diff() reads all fields"
    )

    two = replace(full, retrieve=False, summarize=False)
    assert set(full.diff(two)) == {"retrieve", "summarize"}
    print("switches:", full.switch_map())
    print("one-field and multi-field diffs both detected")
