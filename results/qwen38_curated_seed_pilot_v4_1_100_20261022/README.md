# Curated seed pilot v4.1

This is an experimental 100-record regeneration pilot. It does not update the reviewed training set or production prompt.

The seed pool contains targets selected from the 300-record blinded audit where the prior source review was PASS, pair review was PASS, and no critical issue was flagged. It excludes parenthetical/bracketed editorial text, credential-shaped assignments, and targets outside 20–180 characters. The audit reviewer was ChatGPT, not a human gold annotator.

The v4.1 prompt asks for one controlled oral phenomenon without unrelated names, entities, or facts. At generation time, the validator checked digit strings, ASCII spans, URLs, assignments, shell operators, quoted ASCII literals, and new brackets. A later review found that it missed a number-unit rewrite (`60帧每秒` → `60fps`); a posthoc number-unit validator now catches that case. The generation-time hard check did not include unit spans, and surface checks do not replace semantic review.

`seed_pool_100.jsonl` contains the selected targets and prior oral outputs for later paired comparison. The generator request only sends the scene, phenomenon, and clean target; it does not send the prior output or audit labels.

The first v4.1 sampler also assigned the mixed-code category independently of its ASCII-bearing seed selection. All outputs with that tag are held from category-rate estimates and the strict pilot list.
