# Scoring a run

From the installed release, pass the run directory from the evaluator to the scorer. It checks the recorded SHA-256 of every scored result. Save the output alongside the run if you intend to compare methods.

```bash
python -m benchmark.paper_metrics "$AGOS_OUTPUT_ROOT/smoke" > "$AGOS_OUTPUT_ROOT/smoke/metrics.json"
```

`requested` is the number of tasks selected in the run manifest. `scored` counts successful episodes and valid method failures. `pending` lists tasks without a valid terminal result, including infrastructure failures. The scorer returns status `2` for incomplete coverage and emits the provisional numbers on standard output. Use the evaluator's resume command before reporting a complete split.

| Output | Unit | What is averaged |
| --- | --- | --- |
| `SR` | % | Correct confirmations per scored episode. |
| `OSR` | % | Episodes where the UGV reached a legal opportunity to confirm before a decision. |
| `SPL` | % | Success weighted by the efficiency of the UGV's executed path relative to its oracle path. |
| `NE` | m | Final horizontal distance between the UGV and the deployed target. |
| `ASE` | % | First legal aerial observation, weighted by its UAV decision index within the 80-step budget. |
| `ACG` | % | Candidate guidance quality, discounted by nomination order and location error. |
| `GVA` | % | Correct executed confirm/reject judgments across labeled events. The output includes TP, TN, FP, FN and the number of supported events. |
| `DS` | calls/episode | Mean MLLM decision calls over both roles. |
| `DSR` | % | Mean of success divided by the number of decision calls, with the denominator clamped to one. |

A `null` metric indicates missing support or an inapplicable measurement. For the model-free oracle, DS and DSR are inapplicable. Compare agents on the same split with full coverage, and report event support alongside GVA.
