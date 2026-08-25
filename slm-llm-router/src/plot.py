"""Draw the deferral curve.

Inputs:  the dict from src/evaluate.build_rows
Outputs: output/deferral_curve.png

One chart carries the whole argument. The x-axis is the fraction of traffic
sent to the big model, which with a free local tier is exactly `1 - savings`.
The straight line is what random deferral achieves. Any router worth its
complexity has to sit above that line, and no router can sit above the oracle.

Plotting it this way makes the headline unfakeable: moving the threshold slides
a point ALONG the curve, it never lifts the curve off the chord.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from src.config import CURVE_PATH  # noqa: E402

INK = "#1f2933"
CASCADE = "#4f46e5"
RANDOM = "#9aa5b1"
ORACLE = "#0f9d58"


def draw(data: dict) -> None:
    fig, ax = plt.subplots(figsize=(9, 5.5), dpi=160)

    rand_x = [p["llm_rate"] for p in data["random_curve"]]
    rand_y = [p["accuracy"] for p in data["random_curve"]]
    ax.plot(rand_x, rand_y, color=RANDOM, lw=2, ls="--",
            label="random deferral (the chord)", zorder=2)

    orc_x = [p["llm_rate"] for p in data["oracle_curve"]]
    orc_y = [p["accuracy"] for p in data["oracle_curve"]]
    ax.plot(orc_x, orc_y, color=ORACLE, lw=2, ls=":",
            label="oracle (needs the answer key)", zorder=2)

    cur_x = [p["llm_rate"] for p in data["curve"]]
    cur_y = [p["accuracy"] for p in data["curve"]]
    ax.plot(cur_x, cur_y, color=CASCADE, lw=2.6, marker="o", ms=6,
            label="cascade (local verifier)", zorder=4)

    # Shade what the router actually bought over random deferral.
    ax.fill_between(
        cur_x, cur_y,
        [data["chord"]["slm"][1] + x * (data["chord"]["llm"][1] - data["chord"]["slm"][1])
         for x in cur_x],
        color=CASCADE, alpha=0.10, zorder=1,
    )

    for point in data["curve"]:
        ax.annotate(
            f"s{point['strictness']}",
            (point["llm_rate"], point["accuracy"]),
            textcoords="offset points", xytext=(0, -14),
            ha="center", fontsize=8, color=CASCADE,
        )

    headline = data["headline"]
    ax.scatter([headline["llm_rate"]], [headline["accuracy"]],
               s=170, facecolor="none", edgecolor=CASCADE, lw=2.2, zorder=5)
    ax.annotate(
        f"headline: {headline['savings']:.0%} cheaper\n"
        f"at {headline['accuracy']:.0%} accuracy",
        (headline["llm_rate"], headline["accuracy"]),
        textcoords="offset points", xytext=(18, 14), fontsize=9,
        color=INK,
        arrowprops={"arrowstyle": "-", "color": INK, "lw": 0.8},
    )

    ax.set_xlabel("fraction of queries sent to the large model  "
                  "(with a free local tier, savings = 1 − this)")
    ax.set_ylabel("accuracy")
    ax.set_title(
        f"Deferral curve · {data['n']} items · "
        "a router must bow above the chord to be worth building",
        fontsize=11,
    )
    ax.set_xlim(-0.02, 1.02)
    ax.grid(alpha=0.25, lw=0.6)
    ax.spines[["top", "right"]].set_visible(False)
    ax.legend(frameon=False, loc="lower right", fontsize=9)

    fig.tight_layout()
    fig.savefig(CURVE_PATH)
    plt.close(fig)
    print(f"wrote {CURVE_PATH}")


if __name__ == "__main__":
    from src.evaluate import build_rows
    from src.matrix import load

    draw(build_rows(load()))
