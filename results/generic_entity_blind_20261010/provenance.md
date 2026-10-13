# Generic entity blind-index preparation

This directory contains a deterministic 128-item reference index: 64 Common Voice zh-CN test items and 64 FLEURS zh-CN test items. No ASR output or recognition errors were read or used to build the vocabularies, assign labels, or select examples.

## Sources and audio access

- Common Voice: `/home/aim0/data/datasets/ASR/commonvoice_zh-CN/cv-corpus-26.0-2026-06-12/zh-CN/test.tsv`; sample locations include the TSV physical line, sentence ID, and relative clip path. All 64 selected clip files were checked and exist. Metadata includes only a salted SHA-256 speaker grouping hash, never `client_id`.
- FLEURS: `/home/aim0/data/datasets/ASR/fleurs/zh-CN_test/data-00000-of-00002.arrow` and `data-00001-of-00002.arrow`; metadata preserves shard, zero-based row index, dataset row ID, and audio path stem. Selected waveform bytes are embedded in the Arrow `audio.bytes` field; all 64 selected records were checked and contain non-empty bytes. FLEURS records do not provide a speaker ID, so speaker independence cannot be established.

Exact duplicate reference sentences were collapsed before sampling in both sources. When a duplicate had multiple recordings, the retained recording was the one with the lowest deterministic selection hash of its unique sample ID.

## Vocabulary and sampling

The source-specific glossaries were manually seeded by inspecting reference text and then restricted to complete 2–24-character entity or technical-term strings attested in that source. Terms are matched by exact substring presence in the reference. Each source contributes 48 `lexicon_present` items and 16 `lexicon_absent` items. “Absent” means none of that source's glossary terms appears in the reference; it does not mean the sentence contains no proper names or terminology.

The glossary is fixed before the sample lists are created. Within each source and label stratum, items are sorted by `SHA-256("entity-blind-v1:" + sample_id)`; the first 48 positives and first 16 glossary-absent negatives are retained. The final displayed sample order uses the same hash. The `stats` object in `metadata.json` records eligible and unselected counts.

This is a known-vocabulary entity evaluation: terms were prepared from the same source references, so it measures behavior on a predeclared vocabulary and is not a fully blind test of vocabulary generalization. FLEURS has no speaker ID, and this index cannot establish speaker independence.

## Files

- `metadata.json`: selected sample references, terms present, labels, stable source locations, and selection statistics.
- `glossaries.json`: frozen source-specific term lists. The coarse `entity_or_term` tag is informational; no type-specific rule affects selection.
