# Medical-QA Extension Study

Generated: `2026-07-19T09:12:48.492564+00:00`
Sample: 2773 questions
Completed extension calls: 1000/5046
Observed extension cost: `$0.6357`
Total API charges including duplicate audit rows: `$0.6357`

## Main results

- Reused paper-faithful Qwen baseline accuracy: 86.7%.
- Fully position-invariant across baseline and every rotation: NA.
- Every new rotation answered correctly: NA.
- Candidate-verification exact question accuracy: NA.
- Correct-candidate sensitivity: NA.
- Distractor specificity: NA.

## Variant-level metrics

| Variant | N | Accuracy | Abstention | Strict generative | Option recovery | Question token F1 |
|---|---:|---:|---:|---:|---:|---:|
| choices_to_question | 432 | NA | 0.0% | NA | NA | 0.182 |
| generative_no_choices | 568 | 52.6% | 0.0% | 32.2% | 51.8% | NA |
