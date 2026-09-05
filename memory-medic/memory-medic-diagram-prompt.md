# Architecture Diagram — Image Prompts

Prompts for generating the memory-medic architecture figure so it reads as a
figure lifted from a systems or temporal-database paper, not a marketing
graphic.

---

## Read this first

Image models render *shapes* well and *text* badly. Every one of these prompts
will produce a convincing figure with at least one garbled label, and a diagram
with "MEMROY STORE" in it is worse than no diagram in a paid PDF.

Two ways to handle it:

- **Keep labels short and ALL CAPS, one or two words.** Long labels garble
  first. Every label below is already written this way.
- **Generate, then fix the text.** Take the winner into any vector or slide
  tool and retype the labels over the generated boxes. Five minutes, and it is
  the only way to be certain.

If you need the labels guaranteed correct with no retouching, the Mermaid block
in section 2 of the guide already renders a clean version — it just looks like
a Mermaid diagram rather than a paper figure.

---

## What makes a figure look academic

The levers, worth knowing before you tweak these:

- **Greyscale, or one accent colour at most.** Papers get printed. Heavy colour
  instantly reads as a blog post.
- **Hairline strokes,** 0.5pt to 1pt. Thick outlines look like slideware.
- **Plain rectangles, sharp corners.** Rounded corners are a UI convention.
- **Cylinders for data stores**, never rectangles.
- **Circled step numbers** ① ② ③ on the edges, referenced from the body text.
- **A caption underneath**, serif, beginning "Figure 1:".
- **Tight and dense.** Paper figures fit a two-column layout — compact, small
  type, not airy.

The signature element for *this* project is the horizontal divider separating
the two paths. A security paper has its dashed trust boundary; this project's
whole argument is that there is a second path below the line that nothing else
in the literature draws.

---

## Variant A — Systems paper (recommended)

Two paths converging on one writer. This is the architecture figure.

```
A black and white technical figure from a computer systems research paper, drawn in the style of a LaTeX TikZ diagram. Pure white background, hairline 0.75pt black strokes, no colour, no gradients, no shadows, no 3D, no glow, flat line art.

Composition: a compact horizontal dataflow diagram occupying a two-column paper figure width, dense and tightly spaced with small type, divided into an upper band and a lower band by a single long horizontal dashed line running the full width.

In the UPPER band, left to right, a chain of plain sharp-cornered rectangles connected by thin straight arrows with small solid triangular arrowheads: "MESSAGE", then "EXTRACT", then "CLASSIFY", then "POLICY". Small italic text at the far left of this band reads "arrival".

In the LOWER band, left to right: a cylinder database shape labelled "SOURCES", then a rectangle "DETECT", then a rectangle "PROPOSE". A small clock icon sits to the left of "DETECT". Small italic text at the far left of this band reads "silent decay".

Both bands converge with arrows into a single tall rectangle at the centre-right drawn with a doubled outline, labelled "REPAIR", with tiny italic text beneath it reading "only writer".

From "REPAIR" a solid arrow points right into a cylinder database shape labelled "STORE". From "STORE" a solid arrow points right to a rectangle "RECALL", and from "RECALL" a final arrow to a rectangle "ANSWER".

From "REPAIR" a dashed arrow branches downward to a rectangle labelled "GATE", annotated with tiny italic text "low confidence", and from "GATE" a dashed arrow points to a rectangle labelled "INBOX". A dashed arrow returns from "INBOX" up to "REPAIR".

Small circled step numbers 1 through 5 sit on the arrows entering EXTRACT, POLICY, DETECT, REPAIR and GATE.

Beneath the entire diagram, a caption line in serif italic type: "Figure 1: Two paths into one writer."

Typography: Computer Modern and Helvetica, small, crisp, black. Precise technical draftsmanship, high legibility, academic restraint.

--ar 16:9 --style raw --v 7
```

---

## Variant B — The bitemporal interval figure

The more distinctive figure, and the one that actually explains the idea. This
is a temporal-database plot, not a dataflow diagram — closest to a Snodgrass
temporal-DB paper. Use it as Figure 2 alongside Variant A.

```
A black and white technical figure from a database systems research paper, in the style of a LaTeX TikZ timeline plot. Pure white background, hairline black strokes, no colour, no shading, no 3D, flat line art.

A single horizontal axis runs across the bottom, labelled at its right end in small italic serif text "valid time", with five short tick marks labelled "FEB", "APR", "JUN", "AUG", "OCT".

Above the axis, two horizontal bars drawn as long thin open rectangles, stacked on separate rows.

The lower bar spans from the FEB tick to the JUN tick, is closed at both ends with short vertical end caps, and is labelled inside in small capitals "PUNE". Small italic text above its right end reads "superseded".

The upper bar begins at the JUN tick and extends past the right edge of the plot, ending in an open arrowhead rather than a cap, and is labelled inside in small capitals "BENGALURU".

A single vertical dashed line rises from the axis at a point between AUG and OCT, crossing only the upper bar, labelled at its top in small italic text "query at T".

Two small hollow circular markers sit below the axis with thin leader lines pointing up to the left end of each bar, labelled in tiny italic text "recorded" and "recorded".

Beneath the entire figure, a caption line in serif italic type: "Figure 2: A superseded fact keeps the window it was true for."

Typography: Computer Modern, small, crisp, black. Precise, sparse, academic.

--ar 16:9 --style raw --v 7
```

---

## Variant C — Machine learning paper

Softer, one accent colour, no dashed divider. Closer to a NeurIPS or ICML
systems figure. Use when the greyscale versions feel too austere next to the
rest of the deck.

```
A technical architecture figure from a machine learning research paper, flat 2D vector line art, white background, thin 1pt dark grey strokes, a single muted indigo accent used only for fills, no gradients, no shadows, no 3D.

A horizontal pipeline of plain rectangles with sharp corners connected by thin straight arrows with small solid arrowheads, tightly spaced and compact.

Upper row, left to right: "MESSAGE", "EXTRACT", "CLASSIFY", "POLICY".

Lower row, left to right: a cylinder "SOURCES", then "DETECT", then "PROPOSE".

Both rows feed by arrows into one rectangle "REPAIR", filled with the indigo accent, the only filled shape in the figure. An arrow from "REPAIR" leads to a cylinder "STORE", then to "RECALL", then to "ANSWER".

A dashed arrow drops from "REPAIR" to a rectangle "GATE" and on to "INBOX", with a dashed arrow returning to "REPAIR".

Beneath the figure, a caption in serif italic: "Figure 1: Two paths into one writer."

Typography: Helvetica or Inter, small, dark grey, high legibility. Clean, restrained, publication quality.

--ar 16:9 --style raw --v 7
```

---

## Tweaks that usually help

- **Boxes merging into each other:** add "generous white space between shapes,
  each rectangle clearly separated".
- **Too decorative:** add "no icons, no illustrations, no human figures,
  schematic only" and repeat "flat line art" once more.
- **Arrows curving:** add "all arrows perfectly straight and orthogonal".
- **Labels drifting outside their boxes:** add "every label centred inside its
  shape".
- **Too airy:** add "dense two-column paper figure, small type, compact
  layout".

For DALL·E or Imagen, drop the `--ar 16:9 --style raw --v 7` suffix and ask for
a 16:9 image in the prompt body instead.
