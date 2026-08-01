# Medical-QA Extension Study

Generated: `2026-07-19T08:54:31.115770+00:00`
Sample: 2773 questions
Completed extension calls: 1200/5046
Observed extension cost: `$0.4103`
Total API charges including duplicate audit rows: `$0.5110`

## Main results

- Reused paper-faithful Qwen baseline accuracy: 77.5%.
- Fully position-invariant across baseline and every rotation: NA.
- Every new rotation answered correctly: NA.
- Candidate-verification exact question accuracy: NA.
- Correct-candidate sensitivity: NA.
- Distractor specificity: NA.

## Variant-level metrics

| Variant | N | Accuracy | Abstention | Strict generative | Option recovery | Question token F1 |
|---|---:|---:|---:|---:|---:|---:|
| choices_to_question | 545 | NA | 0.0% | NA | NA | 0.176 |
| generative_no_choices | 655 | 50.1% | 0.0% | 27.6% | 49.8% | NA |
