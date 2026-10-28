# Curated seed pilot v4

This is an experimental 100-record regeneration pilot. It does not update the reviewed training set or production prompt.

The seed pool contains targets selected from the 300-record blinded audit where the prior source review was PASS, pair review was PASS, and no critical issue was flagged. It excludes parenthetical/bracketed editorial text, credential-shaped assignments, and targets outside 20–180 characters. The audit reviewer was ChatGPT, not a human gold annotator.

The v4 generation prompt asks for one controlled oral phenomenon without unrelated names, entities, or facts. The validator checks exact number surfaces, ASCII spans, URLs, assignments, shell operators, quoted ASCII literals, and newly introduced brackets. These checks do not replace semantic review.

`seed_pool_100.jsonl` contains the selected targets and prior oral outputs for later paired comparison. The generator request only sends the scene, phenomenon, and clean target; it does not send the prior output or audit labels.
