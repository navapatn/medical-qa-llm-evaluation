# Medical-QA Extension Study

Generated: `2026-07-19T01:30:23.097609+00:00`
Sample: 2773 questions
Completed extension calls: 5046/5046
Observed extension cost: `$0.8949`
Total API charges including duplicate audit rows: `$0.8953`

## Main results

- Reused paper-faithful Qwen baseline accuracy: 84.5%.
- Fully position-invariant across baseline and every rotation: NA.
- Every new rotation answered correctly: NA.
- Candidate-verification exact question accuracy: NA.
- Correct-candidate sensitivity: NA.
- Distractor specificity: NA.

## Variant-level metrics

| Variant | N | Accuracy | Abstention | Strict generative | Option recovery | Question token F1 |
|---|---:|---:|---:|---:|---:|---:|
| choices_to_question | 2273 | NA | 0.0% | NA | NA | 0.194 |
| generative_no_choices | 2773 | 49.0% | 0.0% | 36.0% | 47.1% | NA |
