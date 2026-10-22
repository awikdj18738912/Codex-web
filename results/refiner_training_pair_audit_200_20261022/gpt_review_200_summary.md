# 200条训练对 GPT 抽样审计

## 结论

这是按风险类型分层抽取的诊断样本，不是简单随机样本，因此下面比例不能外推为全量数据集错误率。审核同时对照口语原文、Refiner 输入和精修目标。

| 判定 | 数量 | 占样本比例 |
|---|---:|---:|
| PASS | 174 | 87.0% |
| MINOR | 10 | 5.0% |
| REVIEW | 6 | 3.0% |
| FAIL | 10 | 5.0% |

确认了明确标签错误，主要包括保留说话者已撤回的候选内容、英文错误信息/专名被改写、实验结果被强化，以及订单号数字被改错。该结果说明数据质量可能是当前精修效果不佳的一个原因，但这份有意覆盖高风险类别的样本不能证明它是唯一或主要原因。

## 需要优先处理的明确错误

- #19 `vibe_coding_11874`：将明确的英文错误文本“Session expired”和“EREMOTE: remote operation failed”翻成中文，违反错误码/英文文本须原样保留的要求。
- #62 `academic_2032`：口语最后明确排除Flickr-Wiki、gradient checkpointing和Neural Radiance Fields；target仍保留前两类候选的铺陈，且将英文数据集名Flickr-Wiki写成Flick-Wiki。
- #66 `vibe_coding_9001`：来源明确说不需要清理临时文件，但target仍保留已放弃的删除命令find /tmp -type f -mtime +7 -delete。
- #87 `daily_chat_12543`：来源明确纠正科举体验套装只是游客体验、并非真能考进士；target仍保留“好像真能考个进士似的”的误导说法。
- #90 `explanation_5535`：来源明确纠正目标是中路藏寨碉楼群、不是中国航天科技集团五院西安中心；target仍复述该明确废弃候选。
- #124 `academic_7312`：源文明确表示量子机器学习部分“根本没想清楚”，并最终收束为自组织临界性重点；target仍保留量子机器学习与PDE残差学习存在相通性的结论，留下被撤回/未定内容。
- #158 `academic_5936`：原文仅说传感数据“挺有用”、整体效果“还行”，target新增“带来了明显提升”，凭空强化了实验结论。
- #169 `customer_service_7192`：原文明确纠正“不是B站”，最终说联系南京云动科技有限公司售后；target仍断言鼠标购自B站会员购并有订单号，保留了明确废弃候选。
- #170 `vibe_coding_5776`：target把 EADDRINUSE、Stack overflow、authToken 分别翻译/改写成中文描述或小写 authToken，违反错误码与英文实体需原样保留的要求。
- #192 `customer_service_3463`：订单号被改成20241124113579246；原文有效订单号为2024112413579246，target多插入了数字，编号不一致。

## 需人工判定

- #65 `academic_6626`：来源称一个完全相同的Hausdorff公式先是错误、后又是正确，无法从文本判定公式是否应改；target照录该矛盾。
- #83 `navigation_852`：来源的路线描述为“到察布查尔站，再从察布查尔到察布查尔站”，起终点重复且随后又询问临夏中转；target改成“前往”仍未给出目的地，无法确定完整路线意图。
- #101 `explanation_9427`：源文多处用同一字面“中煤”构造相反纠正，且拼音段落自相矛盾，无法可靠判定相关标签意图。
- #136 `explanation_11864`：源文对“网”字的多个区分全部同字同音，否定关系无法由文本判定。
- #150 `explanation_22791`：源文“不是陆景琛，而是陆景琛”字面完全相同，纠正对象无法从文本判定。
- #182 `meeting_10408`：原文先将覆盖率99.2与2024-04-22日志相连，随后又自我更正到2024-10-15模拟演练但只明确给出缺货率0.7；无法确定覆盖率是否也应归到10-15，target擅自合并到该日期。

## 范围说明

- 样本来自 `review_pass` 140 条、`repair_accepted` 60 条，生成来源均为 `upstream-teacher-v1`。
- 分层包含 multi_correction、english_entity、numeric_surface、clean_or_minimal、long_text 和 scene_balanced_random。
- 主审对两条边界项做了降级：英文名称大小写格式变化、普通技术词 `case` 省略，保留为 MINOR，不计入明确语义 FAIL；原始分段判定保存在 `reviewer_verdict` 字段。
- 本轮只产出审计结果，没有改动训练集。

## 文件

- `gpt_review_200.jsonl`：完整逐条审计。
- `gpt_review_200.csv`：可筛选文本表。
- `gpt_review_200.xlsx`：含审核结果和汇总页。
- `gpt_review_200_report.json`：计数、抽样构成和 ID 汇总。
