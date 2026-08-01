# Medical-QA Extension Study

Generated: `2026-07-19T07:58:54.854106+00:00`
Sample: 2773 questions
Completed extension calls: 400/5046
Observed extension cost: `$1.9234`
Total API charges including duplicate audit rows: `$2.4499`

## Main results

- Reused paper-faithful Qwen baseline accuracy: 88.4%.
- Fully position-invariant across baseline and every rotation: NA.
- Every new rotation answered correctly: NA.
- Candidate-verification exact question accuracy: NA.
- Correct-candidate sensitivity: NA.
- Distractor specificity: NA.

## Variant-level metrics

| Variant | N | Accuracy | Abstention | Strict generative | Option recovery | Question token F1 |
|---|---:|---:|---:|---:|---:|---:|
| choices_to_question | 199 | NA | 0.0% | NA | NA | 0.237 |
| generative_no_choices | 201 | 58.2% | 0.0% | 32.3% | 58.2% | NA |
