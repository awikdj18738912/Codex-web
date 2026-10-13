# Entity profile quality specification

This schema records only facts present in the supplied background cache. It is a development aid, not a source of audio truth.

Each profile contains:

- `canonical_name`: the cache's profile key, without resolving alternate spellings or same-name people.
- `description`: the cache's own bounded description, retaining its ambiguity caveats.
- `sources`: source identifiers, titles, and URL(s) copied from the cache.
- `profile_level`: `official_source_verified_background` means the cache marks the cited background as verified against an official source. It does not certify a complete identity or every possible attribute.
- `identity_complete`: copied from `complete_identity_profile`; false means identity resolution is incomplete.
- `known_conflicts`: only explicit conflicting or potentially erroneous context recorded in the cache. These are evidence notes, not resolved corrections.
- `usable_for_explicit_attribute_contradiction_review`: whether this record has enough identity and attribute support to participate in an explicit contradiction review. All entries are false because the cache marks every identity incomplete, and it says official background does not establish what a speaker said.

An omitted attribute is unknown, never a negative claim. No other recording's reference is authoritative biography evidence. A profile must not automatically veto an ASR result based on name mismatch, absent biographical details, or these records alone. Any future use for contradiction review requires a sufficiently resolved identity and an explicit, source-supported attribute that is directly contradicted; this cache does not currently meet that bar.
