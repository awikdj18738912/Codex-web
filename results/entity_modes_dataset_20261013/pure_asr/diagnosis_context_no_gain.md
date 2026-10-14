# Why glossary context showed little aggregate gain

Read-only comparison of the completed pure-ASR summary and per-ID details against the frozen parent library metadata/summary. Scores below use the same normalized canonical occurrence rule as the scorer; no inference was run and no reference was used to alter or select a prompt. The protected unseen set was not read.

## Aggregate and paired changes

| Stage | Exact gold hits / 2,860 mentions | Difference from prior stage |
|---|---:|---:|
| Pure ASR raw | 2,543 | — |
| Context raw | 2,545 | +2 net mentions |
| Context clean | 2,543 | −2 from context raw; equal aggregate total to pure |

The `pure → context_raw` per-record hit deltas are: 23 recordings at −1 mention, 23 at +1, one at +2, and 2,459 unchanged. That is 25 gained versus 23 lost mentions, for only +2 net. The two domains outside news that contribute this net gain are medical (+1) and natural science (+1); every other non-news library has zero net change. News itself has no raw net gain.

Context clean does not retain those two net mentions. It has no per-record hit gains over context raw, and loses one mention on each of two news recordings:

| ID | Gold lost | Reference | Context raw | Context clean | Hits: pure / raw / clean |
|---|---|---|---|---|---:|
| leb1_479266c091f6591607bf | ORG 宁波市三三改一拆办 | 记者和宁波市三三改一拆办工作人员前往现场 | 记者和宁波市三三改一拆办工作人员前往现场。 | 记者和宁波市三改一拆办工作人员前往现场。 | 1 / 1 / 0 |
| leb1_72b14d2d277d6ea72564 | ORG 温岭市一一零指挥中心 | 浙江台州温岭市一一零指挥中心接警称 | 浙江台州温岭市。一一零指挥中心接警称。 | 浙江台州温岭市。一零指挥中心接警称。 | 3 / 3 / 2 |

In each row, clean output collapses a repeated “三”/“一” sequence that appears in the gold canonical, removing its exact mention hit. This identifies the observed transcript transition; it does not establish whether those sequences are acoustically correct.

## How much of the gold fits in context16?

Across the full set, 193/2,860 (6.75%) annotated mentions match the fixed context16 terms. Those targets receive 187 hits in pure raw, 190 in context raw, and 190 in context clean. The remaining 2,667 mentions receive 2,356, 2,355, and 2,353 hits, respectively. Thus context improves prompt-covered mentions by three, while outside-prompt mentions fall by one in raw and three in clean. These parts sum to the aggregate deltas above.

For the news library, the 16 context terms cover only 17/2,445 mentions (0.70%) and 16/1,051 distinct gold names (1.52%). Prompt-covered news mentions score 16/17 in pure raw and 17/17 in each context stage, a one-mention gain. Outside those terms, news scores are 2,127/2,428 in pure, 2,126/2,428 in context raw, and 2,124/2,428 in context clean. The per-record raw gains (23 mentions across 22 samples) exactly balance 23 lost mentions across 23 samples, leaving news at 2,143 total hits in both pure and context raw. The full news library contains 1,051 entries, but its context list includes only these first 16 frozen terms:

`一汽大众`, `丁香湖公园`, `七五村`, `三江源`, `三藩市纪事报`, `上海市公积金管理中心`, `上海易居房地产研究院`, `上海易居研究所`, `上集镇`, `上集镇派出所`, `世界反兴奋剂机构`, `世界贸易组织`, `世纪互联`, `东地产财经`, `东地产财经周刊`, `东山派出所`.

The context list is built once from the frozen selected library and reused for each sample assigned to that library. The audit did not choose terms by the current clip’s gold. The broader evaluation retains its pre-known vocabulary convention because the original frozen glossary itself was curated from reference-supported terms.

## Gold that the short-term filter cannot prompt

There are 1,200 gold mention occurrences across 358 distinct canonical names with length below three characters; all 1,200 match an enabled entry in the corresponding frozen library. The news library contains 1,131 such occurrences across 303 distinct names. `build_entity_asr_context` excludes every one of these candidates before selecting context16 because its short-term rule requires length at least three. Frequent news examples among those excluded candidates include 苹果 (116 occurrences), 北京 (89), 温州 (77), 朱婷 (45), 美的 (28), and 湖北 (27). These are examples of the coverage ceiling, not recommendations to prompt ambiguous short strings automatically.

For news, the short-term filter excludes 46.3% of the 2,445 gold mention occurrences before context16 selection; the remaining longer names still compete for a maximum of 16 terms. This explains why even the known-vocabulary context condition covers only 17 news mentions. It is a structural prompt-capacity limitation, not evidence that all omitted terms would help if added.

## Interpretation limits

- Aggregate equality of pure and context-clean hits (2,543 each) does not mean identical transcripts: 762 IDs differ in text; 49 IDs differ in gold hit count (24 context-clean higher, 25 pure higher), with differences that net to zero.
- Context raw’s +2 is a net across samples and domains, not a broad per-clip benefit. In the dominant news domain it has 23 mention gains and 23 losses.
- Context16 coverage and short-term exclusions are static overlaps with the frozen reference-derived library. They do not estimate causal prompt effect by themselves and do not establish that omitted gold terms are safe or useful prompts.
- CER also reflects numeric and formatting normalization. It should not be used alone to claim perceived quality changed; the entity-hit metric is exact canonical surface matching and does not resolve aliases or real-world identity.
