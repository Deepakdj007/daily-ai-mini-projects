# Architecture Diagram — Image Prompts

Prompts for generating the guardrails-layer architecture figure so it reads as
a figure lifted from a security paper, not as a marketing graphic.

---

## What actually makes a figure look academic

Worth knowing before you tweak these, because these are the levers:

- **Greyscale, or one accent colour at most.** Papers are printed. Figures are
  designed to survive black-and-white printing, so heavy colour instantly
  reads as a blog post.
- **Hairline strokes.** 0.5pt to 1pt. Thick outlines look like slideware.
- **Plain rectangles, not rounded ones.** Rounded corners are a UI convention.
- **Dashed trust boundaries.** This is *the* signature element of a security
  paper figure — a dashed box drawn around the untrusted region.
- **Circled step numbers** ① ② ③ on the edges, referenced from the body text.
- **A caption underneath**, serif, beginning "Figure 1:".
- **Cylinders for data stores**, never rectangles.
- **Tight and dense.** Paper figures fit a two-column layout, so they are
  compact with small type, not airy.

The trust boundary matters most for this project, because the whole argument is
about which side of that line each layer sits on.

---

## Variant A — Security paper (recommended)

Greyscale, trust boundary, circled steps. Closest to USENIX Security / IEEE S&P.

```
A black and white technical figure from a computer security research paper, drawn in the style of a LaTeX TikZ diagram. Pure white background, hairline 0.75pt black strokes, no colour, no gradients, no shadows, no 3D, no glow, flat line art.

Composition: a compact horizontal dataflow diagram occupying a two-column paper figure width, dense and tightly spaced with small type.

A large dashed rounded rectangle on the left encloses two elements and is labelled in small italic text at its top edge "untrusted". Inside it: a small plain rectangle labelled "USER" and a cylinder database shape labelled "DOCS".

Outside and to the right of the dashed boundary, a horizontal chain of plain sharp-cornered rectangles connected by thin straight arrows with small solid triangular arrowheads, in this order: "NORMALIZE", "CLASSIFIER", "MODEL", "PII FILTER", "CANARY", and a final rectangle "ANSWER".

Above the chain, offset upward, a rectangle labelled "JUDGE" connected to "CLASSIFIER" by a branching arrow annotated with tiny italic text "p > 0.5". A short arrow from "JUDGE" points down-left to a small rectangle labelled "REFUSE".

A second vertical dashed line runs between "MODEL" and "PII FILTER", labelled in small italic rotated text "output side".

Small circled step numbers 1 through 5 sit on the arrows entering NORMALIZE, CLASSIFIER, JUDGE, PII FILTER and CANARY.

Beneath the entire diagram, a caption line in serif italic type: "Figure 1: Five-layer guardrail pipeline."

Typography: Computer Modern and Helvetica, small, crisp, black. Precise technical draftsmanship, high legibility, academic restraint.

--ar 16:9 --style raw --v 7
```

---

## Variant B — Machine learning paper

Slightly softer, one accent colour, no trust boundary. Closer to a NeurIPS or
ICML systems figure.

```
A technical architecture figure from a machine learning research paper, flat 2D vector line art, white background, thin 1pt strokes, no gradients, no shadows, no 3D.

A single horizontal pipeline of plain rectangles with sharp corners, connected by thin straight arrows with small solid arrowheads, tightly spaced and compact.

Left to right: a rectangle "USER" and a cylinder "DOCS" both feeding into "NORMALIZE", then "CLASSIFIER", then "MODEL", then "PII FILTER", then "CANARY", then "ANSWER".

A rectangle "JUDGE" sits above the line, connected to "CLASSIFIER" by a branching arrow with a tiny italic edge label. A short arrow leads from "JUDGE" down to "REFUSE".

Entirely greyscale except for exactly one accent colour, a muted slate blue, used only to fill the three boxes NORMALIZE, CLASSIFIER and JUDGE with a very light 10 percent tint. Every other box is white with a black outline.

Small circled numerals label each stage. A serif italic caption sits below the figure reading "Figure 1: Guardrail layers and their placement relative to the model."

Typography: crisp Helvetica for box labels in small caps, serif for the caption. Dense two-column figure proportions, generous internal alignment, minimal decoration.

--ar 16:9 --style raw --v 7
```

---

## Variant C — Explicit TikZ / printed page

Use when A and B still look too "designed". This one names the rendering
toolchain, which pushes the model hard toward the right aesthetic.

```
A figure rendered in LaTeX TikZ, as printed in an academic journal. Black ink on white paper, hairline rule weights, monochrome, absolutely no colour.

Node-and-edge dataflow diagram: rectangular nodes with thin borders, straight orthogonal connector edges with arrowheads, right-angle bends where edges turn corners.

Nodes left to right: "USER" and a cylinder "DOCS" enclosed together inside a dashed boundary labelled "untrusted", then "NORMALIZE", "CLASSIFIER", "MODEL", "PII FILTER", "CANARY", "ANSWER". A node "JUDGE" sits above the main row with a branch edge from "CLASSIFIER" and an edge down to "REFUSE".

Edge labels in tiny italic serif. Circled numerals on the edges. Caption below in serif: "Figure 1: Guardrail pipeline. Layers 4 and 5 read model output and cannot be overridden by instructions."

Style: pgfplots and TikZ output, IEEE transactions figure, camera-ready academic typesetting, orthogonal routing, precise alignment, dense compact layout, no decoration whatsoever.

--ar 3:2 --style raw --v 7
```

---

## Getting the text right

Every current image model garbles some labels, and a paper figure is mostly
labels. Three options, in order of how reliably they work:

**1. Real TikZ.** For an actual paper, write the figure in TikZ or use
[tikzcd](https://tikzcd.yichuanshen.de/). Guaranteed correct, vector, and
indistinguishable from a real paper figure because it is one.

**2. Mermaid, then style it.** The Mermaid block in the guide already encodes
the correct flow. Paste it into [mermaid.live](https://mermaid.live), export
SVG, then in Figma or Illustrator: switch everything to greyscale, drop stroke
weights to 0.75pt, square off the corners, add the dashed trust boundary and a
serif caption. Twenty minutes, and it beats any generated image.

**3. Generate the layout, typeset the labels yourself.** Replace every label in
the prompts above with "unlabelled empty rectangles", then add text in a vector
editor over the generated layout.

Option 2 is the sweet spot for a carousel or a PDF. Option 1 if this ever goes
into something citable.

---

## Source of truth

Whatever you generate has to match this flow:

```
        ┌─ untrusted ─┐
        │  USER       │
        │  DOCS       │
        └──────┬──────┘
               ↓
        NORMALIZE → CLASSIFIER ──p>0.5──→ JUDGE ──block──→ REFUSE
                         │                  │
                         └──── clean ───────┴── allow ──→ MODEL
                                                            │
                              ANSWER ← CANARY ← PII FILTER ←┘
                                      └── output side ──┘
```

Three properties the figure has to make visible, because they carry the whole
argument:

- **DOCS sit inside the untrusted boundary alongside USER.** Indirect injection
  arrives through retrieval, not through the chat box.
- **JUDGE is a branch, not a stage.** It runs only when the classifier is
  suspicious, which is why the arrow into it carries a condition label.
- **PII FILTER and CANARY sit after MODEL.** They read output, so there is no
  instruction in them to override — that is why they are the only layers whose
  behaviour does not change when you swap the model.
