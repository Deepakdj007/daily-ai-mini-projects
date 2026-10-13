# semantic-cache — diagram image prompt

Attach two images when you paste this: the Chroma logo first, then the Gemini logo.

---

Create a clean, professional architecture diagram as an image, landscape 16:9, on a pure white background. Flat 2D design, like a diagram in a polished engineering blog post, with generous white space and text that's sharp and readable at Instagram size. The first attached image is the Chroma logo and the second is the Gemini logo. Reproduce both exactly as they appear in the attachments, with the same shapes, letters and colours; never redraw, restyle or recolour them.

At the top left, put the title in large bold dark-navy text: "Semantic Cache." Below it, a smaller grey subtitle: "Same meaning, same answer. Pay for the LLM only once." At the top right, on the title's line, place the Chroma and Gemini logos side by side, small.

The diagram runs left to right. It starts with a rounded blue box with a chat-bubble icon labelled "Question". An arrow leads to a teal box with a small vector-arrow icon labelled "Embed", with the small caption "MiniLM, local". An arrow leads to a red box with the Chroma logo labelled "Cache lookup", with the small caption "nearest past question".

From Cache lookup, two arrows fan out to the right, each with a small label on the arrow. The top arrow is labelled in green "similarity ≥ 0.82" and leads to a green box with a lightning-bolt icon labelled "Cached answer", with the small caption "6 ms · $0". The bottom arrow is labelled in orange "miss or stale" and leads to an orange box with the Gemini logo labelled "Call LLM", with the small caption "~900 ms · paid". An arrow leads from Call LLM to a purple box with a database icon labelled "Store", with the small caption "question + answer".

From Cached answer and from Store, arrows converge to the right on a final rounded blue box with a speech-bubble icon labelled "Answer".

Below the Cache lookup box, add a small grey dashed-outline box labelled "Invalidation" with three short lines inside: "TTL", "version bump", "targeted delete". Connect it to Cache lookup with a thin dashed grey line.

Along the bottom, add one thin line of small grey text: "60 questions: 53% cheaper · 32 of 60 from cache · 0 wrong answers".

Every box has a white fill, a coloured outline and black text: blue for Question and Answer, teal for Embed, red for Cache lookup, green for Cached answer, orange for Call LLM, purple for Store, dashed grey for Invalidation. Use thin grey arrows with small arrowheads that never cross a box or a label, simple outline icons in the same colour as their box, and one clean sans-serif font throughout, like Inter. No drop shadows, gradients, 3D, glow, neon, dark backgrounds, people, robots, circuit patterns, clipart or emoji. Spell every word exactly as written here and add no other text.
