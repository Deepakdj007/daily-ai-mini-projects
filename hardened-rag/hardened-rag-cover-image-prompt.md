# hardened-rag — cover image prompts

Two directions. The robot pair reads instantly on a feed and is the better
carousel cover. The archivist is quieter and better for the PDF title page.

---

## Direction A — the sentinel and the thing behind the racks

One metaphor: something is hiding inside your own knowledge base, and only a
machine that checks provenance ever looks behind the shelf.

### Primary

> Cinematic wide shot, 16:9, photoreal. A tall humanoid sentinel robot stands in
> the cold aisle of a vast data centre, mid-turn, head tilted as it scans down a
> corridor of towering server racks. Its body is matte brushed aluminium and dark
> composite — industrial, functional, no chrome, no glow panels. Two narrow
> horizontal eye slits emit a soft green light that falls on the rack in front of
> it like a torch beam, catching fine dust in the air.
>
> Deep in the background, far down the aisle and mostly hidden behind the corner
> of a massive server cabinet, a second robot of the same build crouches in
> shadow. Almost nothing of it is visible — a shoulder, the edge of a head — and
> two small red eyes, the only red anywhere in the frame, catching the light for
> an instant.
>
> Lighting: one hard cold key from high left raking down the aisle, everything
> else falling into near-black. Palette graded teal and steel with a single red
> accent. Shot on anamorphic 40mm, shallow depth of field, the foreground robot
> sharp and the hiding one soft. Fine dust in the light shafts, no smoke, no
> haze machine, no lens flare, no floating UI, no holograms, no text.
>
> Composition: the sentinel occupies the right third, the aisle recedes to the
> left, leaving the upper-left quadrant dark and empty for a headline.

`--ar 16:9 --style raw --v 7`

### Alternate — closer, more tension

> Cinematic medium shot, 16:9, photoreal. Over-the-shoulder from behind a
> sentinel robot's head as it leans around the end of a server rack, green eye
> light spilling across the metal and lighting the dust. In the gap between two
> cabinets, deep in shadow and badly out of focus, two red points of light are
> looking straight back at the camera.
>
> The green is the light doing the searching; the red is the thing that has
> already been found and knows it. Hard single source from above, everything
> else black. Matte industrial materials, no chrome, no glow strips, no neon, no
> holograms, no text. Anamorphic 50mm, heavy bokeh on the background.
>
> Leave the left third dark and uncluttered for a headline.

`--ar 16:9 --style raw --v 7`

### For ChatGPT or DALL·E

Drop the `--ar` flag, ask for landscape in words, and add this after the
description, because these models tend to over-light a dark frame:

> Keep the image genuinely dark. Most of the frame should be in shadow, with one
> hard light source. The red eyes must be small and partially hidden, not a
> glowing beacon. No text anywhere in the image.

---

## Direction B — the archivist (quieter, for the PDF title page)

One metaphor: a document that looks exactly like the real thing, held up
against the light, and only then showing what it is.

> Extreme close-up, cinematic. A woman archivist in her thirties holds a single
> sheet of paper up against a bare window, backlit hard so the page glows and the
> printed text reverses through it. Her eyes are narrowed in the moment of
> recognition. Behind the glowing sheet, a second sheet is visible through it,
> misaligned by a few millimetres, its text almost but not quite matching. Dust
> in the shaft of light. Deep shadow on the room side of her face, warm daylight
> rim on the other. Muted archive-room palette, oxidised paper, dark wood. Shot
> on 85mm, shallow depth of field, natural light only. Photographic, no text
> legible.

`--ar 16:9 --style raw --v 7`

---

## What makes these work, and what breaks them

**One light source.** Both directions live or die on a single hard key with
everything else falling away. Even lighting kills the whole idea, because the
image is about something being concealed.

**Red appears exactly once.** The moment there is red anywhere else — a status
LED, a cable, a rim light — the hiding robot stops being a discovery and becomes
set dressing. Say "the only red in the frame" explicitly; models will add more
if you let them.

**The hidden one must be genuinely hard to see.** If it is centred and clearly
lit, the picture is two robots standing around. It has to be small, occluded,
and soft, so the viewer finds it a second after the sentinel.

**Keep a quadrant empty.** Every one of these leaves a dark corner with nothing
in it. That is where the headline goes, and a beautiful image with no room for
text is not a cover.

Avoid throughout: chrome, glowing chest panels, neon strips, holograms, floating
UI cards, network graphics, circuit-board motifs, smoke machines, lens flare,
and any legible text.
