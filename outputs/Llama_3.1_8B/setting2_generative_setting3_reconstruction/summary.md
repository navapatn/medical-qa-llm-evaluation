# Medical-QA Extension Study

Generated: `2026-07-19T01:32:01.026336+00:00`
Sample: 2773 questions
Completed extension calls: 5045/5046
Observed extension cost: `$0.0384`
Total API charges including duplicate audit rows: `$0.0408`

## Main results

- Reused paper-faithful Qwen baseline accuracy: 66.1%.
- Fully position-invariant across baseline and every rotation: NA.
- Every new rotation answered correctly: NA.
- Candidate-verification exact question accuracy: NA.
- Correct-candidate sensitivity: NA.
- Distractor specificity: NA.

## Variant-level metrics

| Variant | N | Accuracy | Abstention | Strict generative | Option recovery | Question token F1 |
|---|---:|---:|---:|---:|---:|---:|
| choices_to_question | 2273 | NA | 0.0% | NA | NA | 0.199 |
| generative_no_choices | 2772 | 26.3% | 0.0% | 16.8% | 21.5% | NA |
