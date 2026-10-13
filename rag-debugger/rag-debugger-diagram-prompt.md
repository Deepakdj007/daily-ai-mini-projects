# rag-debugger — diagram image prompt

Attach two images when you paste this: the Chroma logo first, then the Groq logo.

---

Create a clean, professional architecture diagram as an image, landscape 16:9, on a pure white background. Flat 2D design, like a diagram in a polished engineering blog post, with generous white space and text that's sharp and readable at Instagram size. The first attached image is the Chroma logo and the second is the Groq logo. Reproduce both exactly as they appear in the attachments, with the same shapes, letters and colours; never redraw, restyle or recolour them.

At the top left, put the title in large bold dark-navy text: "See why retrieval failed." Below it, a smaller grey subtitle: "A judge reads what your retriever found, and names the failure." At the top right, on the title's line, place the Chroma logo and the Groq logo side by side, small.

The diagram has two rows. The top row, labelled in small grey caps "THE PIPELINE", runs left to right. It starts with a rounded blue box with a chat-bubble icon labelled "Question". An arrow leads to a teal box with the Chroma logo labelled "Retriever", with the small caption "top 10". An arrow leads to a teal box with a scissors icon labelled "Keep top 3". An arrow leads to a green box with the Groq logo labelled "Answer".

The bottom row, labelled in small grey caps "THE DEBUGGER", sits under the top row. From the Retriever box, a dashed grey arrow goes down into a red box with the Groq logo and a magnifying-glass icon labelled "LLM judge", with the small caption "reads all 10, sees no ranks". From the judge, an arrow leads right to a 2 by 2 grid of four small square tiles. The top-left tile is green and reads "HIT", the top-right tile is red and reads "NOISE", the bottom-left tile is yellow and reads "MISS", and the bottom-right tile is light grey and reads "CUT". Above the grid, in small grey text, "reached the prompt?" runs along the top edge, and "answers the question?" runs down the left edge. From the grid, an arrow leads right to a purple box with a document icon labelled "Trace JSON", with the small caption "every step timed". A dashed grey arrow also runs down from the "Keep top 3" box into the grid.

Along the bottom, add one thin line of small grey text: "4 verdicts: HEALTHY · NOISY_CONTEXT · RETRIEVAL_MISS · CORPUS_GAP".

Every box has a white fill, a coloured outline and black text: blue for Question, teal for Retriever and Keep top 3, green for Answer, red for LLM judge, purple for Trace JSON. The four grid tiles have a solid pale fill in their own colour and black text. Use thin grey arrows with small arrowheads that never cross a box or a label, simple outline icons in the same colour as their box, and one clean sans-serif font throughout, like Inter. No drop shadows, gradients, 3D, glow, neon, dark backgrounds, people, robots, circuit patterns, clipart or emoji. Spell every word exactly as written here and add no other text.
