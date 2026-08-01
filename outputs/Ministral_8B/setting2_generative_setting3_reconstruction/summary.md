# Medical-QA Extension Study

Generated: `2026-07-18T22:30:46.609037+00:00`
Sample: 2773 questions
Completed extension calls: 5046/5046
Observed extension cost: `$0.2190`
Total API charges including duplicate audit rows: `$0.2193`

## Main results

- Reused paper-faithful Qwen baseline accuracy: 74.3%.
- Fully position-invariant across baseline and every rotation: NA.
- Every new rotation answered correctly: NA.
- Candidate-verification exact question accuracy: NA.
- Correct-candidate sensitivity: NA.
- Distractor specificity: NA.

## Variant-level metrics

| Variant | N | Accuracy | Abstention | Strict generative | Option recovery | Question token F1 |
|---|---:|---:|---:|---:|---:|---:|
| choices_to_question | 2273 | NA | 0.0% | NA | NA | 0.214 |
| generative_no_choices | 2773 | 36.8% | 0.0% | 24.7% | 34.4% | NA |
