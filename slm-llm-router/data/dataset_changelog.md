# Eval set changelog

Every edit to `queries.json` after its first use is recorded here, with the
reason. An eval set that can be quietly edited after seeing results is not a
measurement, so the discipline is: change it, say why, and re-run everything.

## 2026-08-15 — pre-freeze, during the calibration gate

Three classification items were replaced **before** the first full run, after a
15-item calibration probe showed `cls-05` failing on *both* tiers.

| id | was | now | reason |
|---|---|---|---|
| `cls-05` | "Please cancel my subscription before it renews next month." → `billing` | "Please send me a GST invoice for last month's payment." → `billing` | Both tiers answered `account`, and they were right to. Cancelling a subscription is defensibly an account action. The item was testing the author's labelling opinion, not model capability. |
| `cls-06` | "The password reset link never arrives in my inbox." → `account` | "Please delete my account and erase all my saved data." → `account` | Ambiguous between `account` and `technical` — an undelivered email is plausibly a delivery fault. |
| `cls-09` | "Two-factor codes are rejected on my new phone." → `account` | "Search returns no results even for products I know exist." → `technical` | Ambiguous between `account` and `technical`. |

The failure mode being avoided: an item where two capable models agree on an
answer the author marked wrong is measuring the author, and it depresses both
tiers equally while adding noise to every comparison.

No edits have been made since the dataset was frozen.
