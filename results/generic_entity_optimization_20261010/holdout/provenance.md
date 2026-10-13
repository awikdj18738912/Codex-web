# Generic entity optimization holdout index

This directory contains a deterministic holdout reference index prepared from Common Voice zh-CN test and FLEURS zh-CN test. The 128 sample IDs and exact reference strings in `results/generic_entity_blind_20261010/metadata.json` were excluded. Exact duplicate references were collapsed within each source, retaining the record with the lowest SHA-256 hash under the holdout selection seed. No ASR output or recognition errors were used for vocabulary construction, labeling, or sample selection. The established decision threshold is 0.63; this holdout must not be used to select or tune a threshold.

## Vocabulary and selection

`glossaries.json` is copied byte-for-byte from the prior blind-index directory. Terms are matched as exact substrings in each source reference. Common Voice contributes 32 remaining glossary-present records and 16 glossary-absent records. FLEURS contributes every remaining glossary-present record (12) and fills the sample to 48 with glossary-absent records (36); the realized pool counts are recorded in `metadata.json`. Items are ordered within each source and label stratum by SHA-256(`entity-gate-holdout-v1:` + sample_id).

## Source locations and speaker information

Common Voice source: `/home/aim0/data/datasets/ASR/commonvoice_zh-CN/cv-corpus-26.0-2026-06-12/zh-CN/test.tsv`. Metadata records TSV physical line, sentence ID, and relative clip path. Every selected clip was checked for filesystem availability. The prior index contains salted speaker hashes, but the salt/algorithm could not be recovered from the recorded metadata, so this holdout leaves `speaker_hash` empty for Common Voice. Speaker independence cannot be established. FLEURS source: `/home/aim0/data/datasets/ASR/fleurs/zh-CN_test/data-00000-of-00002.arrow` and `data-00001-of-00002.arrow`. Metadata records shard, zero-based row index, dataset row ID, and audio path stem. The reference is read from `raw_transcription`, matching the prior index. Every selected embedded audio payload was checked and non-empty. FLEURS provides no speaker ID, so speaker independence cannot be established there either.

No glossary terms or aliases were added.
