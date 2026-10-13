# AISHELL-1 staging pool

This is the full aligned, previously unused AISHELL-1 test recording pool, derived from official AISHELL-NER spans and the official AISHELL-1 reference transcript. No aliases were generated.

- 6,848 aligned unused recordings; 2,013 contain at least one eligible entity.
- 1,108 globally deduplicated eligible canonical names. A name seen under multiple NER types is retained once and marked `CONFLICT`, with all types and per-sample evidence.
- Eligibility: 2–24 Han characters. Explicit generic words are removed by exact match. Short proper names such as 李娜、上海、天津 remain eligible.
- The 328 historically evaluated recording IDs were excluded. Recording IDs and source audio hashes were checked for overlap.
- `start`/`end` are zero-based Unicode code-point offsets into the official reference; end is exclusive. Removing whitespace from the span yields the complete entity spelling because the reference uses spaces between tokens.
- Audio is mono 16 kHz PCM16. `audio_sha256` hashes source WAV bytes; `source_pcm16_sha256` hashes decoded little-endian signed PCM16 samples. `pcm_sha256` follows the evaluation convention: interpret PCM16 with `np.frombuffer(raw, dtype='<i2')`, cast to `<f4`, divide by 32768.0, and hash the resulting float32 little-endian bytes.
- `entity_quality_review.json` lists lightweight generic/boundary review candidates and all cross-type conflicts. These are review flags only; no aliases, corrections, exclusions, or merges are applied.

AISHELL contributes PERSON/LOC/ORG entities from news-domain speech. Its measured distinct-name count does not represent the multi-domain benchmark total.
