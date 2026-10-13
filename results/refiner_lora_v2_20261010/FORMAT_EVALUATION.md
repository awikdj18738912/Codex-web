# Refiner LoRA v2 evaluation

## Scope

This is a controlled, synthetic text-format experiment. The 250-row test split is the same frozen split used for v1 (de151046b1a7b746b8c0250ac3a6c6fcf351a0b2e6effe407919c47c0107d71e). It contains no real ASR audio pairs and no human-confirmed audio gold. Exact-target match is therefore a synthetic format metric, not speech-recognition accuracy or WER.

Both adapters were evaluated with the same base checkpoint, runtime-compatible prompt, frozen test rows, and strict-local patch guard.

## Training

V2 used 816 training rows: the original 628 plus 188 train-only corruptions. The new examples contain 95 character restarts, 72 duplicated safe lexical spans, and 21 filler prefixes. Another 125 candidate rows were skipped because no safe lexical span was available. The original 122-row development set and 250-row test set were copied byte-for-byte.

Training completed three epochs and 306 optimizer steps in 6m33s. Best development loss was 0.09455 at checkpoint 306. The v1 development loss was 0.1625. Despite the lower development loss, v2 performed worse on the frozen test set.

## Frozen-test comparison

| Metric | V1 | V2 |
| --- | ---: | ---: |
| Valid strict-local responses | 91.6% | 80.0% |
| Exact synthetic target | 66.4% | 57.6% |
| Accepted replace actions | 0 | 44 |
| Mean model latency | 470.85 ms | 534.28 ms |

| Test category | V1 exact | V2 exact |
| --- | ---: | ---: |
| Clean | 42/42 | 21/42 |
| Filler | 0/42 | 0/42 |
| Meaningful repetition | 41/41 | 41/41 |
| Protected content | 42/42 | 36/42 |
| Stutter | 0/42 | 5/42 |
| Syntax boundary | 41/41 | 41/41 |

V2 began proposing corrections, but most did not recover the target. Of 44 accepted replacements, 21 changed clean examples, 6 changed protected-content examples, and 17 changed stutter examples; only 5 of those 17 stutter changes matched the target. It overcorrected clean input: for example, “我们将在北京讨论下一步的安排。” became “将在北京讨论下一步的安排。” It preserved meaningful repetitions, but still did not correctly remove any of the 42 filler examples.

## Decision

Do not deploy v2. It has lower exact-target accuracy and lower protocol validity than v1 on the same frozen test set, despite its lower development loss. Keep v1 and v2 as experimental adapters only; leave the production gate and model selection unchanged.

The next useful training data should include diverse, realistic ASR-output/reference pairs. More synthetic text-only variants of this kind are not supported by these results.
