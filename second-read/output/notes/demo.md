# demo - second read

> Every claim below was re-asked with the image removed and with another patient's image in its place. A claim the model still makes without the image is marked as language prior, not evidence. Nothing here is a clinical opinion.

## Findings

The heart size is normal. The lungs are clear. There is no pleural effusion or pneumothorax. Mild degenerative changes of the spine.

## Impression

No acute disease.

## Claim audit

Each cell is the model's own answer to *is this finding present?* under one condition. The only thing that changes between the columns is which image travelled with the question.

| claim | finding | in the note | asked again, with the film | with no film | with a stranger's film | verdict |
|---|---|:--:|:--:|:--:|:--:|---|
| demo-c01 | cardiomegaly, meaning an enlarged cardiac silhouette | yes | yes | yes | no | LANGUAGE PRIOR: said this with the image removed |

**1 of 1 claims came back unchanged when the film was deleted.** Those are language prior, not evidence.
