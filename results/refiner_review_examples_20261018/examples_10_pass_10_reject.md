# 历史原版500条实验：10条通过、10条拒绝

整理时间：2026-10-18 12:07:02 CST

来源：`/home/aim0/桌面/tan/ASR/Codex-web/results/refiner_upstream_500_20261017/qc_results.jsonl`。原实验500条，老师判通过480条、拒绝20条。

**这20条是此前原版500条实验的真实审核记录，不是当前10万条任务的审核结果。当前任务仍在生成Oral，尚未生成Clean或进入语义审核。原版历史实验包含数字处理，且审核提示与当前本地配置不保证完全相同。通过/拒绝是老师当时的判断，并非人工真值。**

当前进度快照：```{"checked_at": "2026-10-18T12:07:02.695733+08:00", "checkpoint": {"generated_count": 3177, "attempted_count": 3560}, "oral_lines": 3177, "recent_rate_records_per_second": 0.2508591065292096, "recent_window": {"from": "2026-10-18 12:01:55+08:00", "to": "2026-10-18 12:06:46+08:00", "added": 73, "seconds": 291.0}, "last_progress_detail": "11:40<12:48:48,  4.08s/item", "semantic_review_started": false}```

## 历史老师审核通过

### 1. daily_chat_5（daily_chat）

**输入（完整原文）**

我昨天路过厦门环岛路的滨海自行车道‘海风长廊’，那里的风景真不错，骑车的时候海风轻轻吹过来，特别舒服，而且听说那边最近还增加了安全护栏和夜间照明，挺贴心的。

**精修结果（完整原文）**

我昨天路过厦门环岛路的滨海自行车道‘海风长廊’，那里的风景很美，骑车时海风轻拂，非常舒适。听说最近还增设了安全护栏和夜间照明，很贴心。

**老师原始审核理由**

The target accurately preserves all key details—location (Xiamen Huan Dao Road's seaside bicycle path 'Hai Feng Long Lang'), scenery, pleasant sea breeze, and recent additions of safety barriers and night lighting—without omission, hallucination, or over-specification. Filler and repetition are removed, and formatting is improved, but the final intended meaning is fully maintained.

### 2. vibe_coding_15（vibe_coding）

**输入（完整原文）**

我刚发现一个内存泄漏的问题，得回滚一下，之前是版本三，改了之后又发现认证失败，最后还是得回到版本三，用 aws s3 cp 把数据传回去，就用 case 这个分支

**精修结果（完整原文）**

我刚发现一个内存泄漏问题，需要回滚到之前版本三，修改后发现认证失败，最终仍需回到版本三，使用aws s3 cp将数据传回，通过case分支

**老师原始审核理由**

All key details preserved: memory leak issue, rollback to version three, failed authentication after change, return to version three, use of aws s3 cp to transfer data, and execution via case branch. No hallucinations, omissions, or over-specific rewrites. Filler and repetition removed, formatting improved. Final intended meaning fully captured.

### 3. explanation_49（explanation）

**输入（完整原文）**

你要是说‘拼多多’，那可不是‘拼多都’，是‘拼’加‘多’再加‘多’，就是那个做团购的电商，名字读起来像‘pīn duō duō’，别搞混了。

**精修结果（完整原文）**

你要是说‘拼多多’，那可不是‘拼多都’，而是‘拼’加‘多’再加‘多’，就是那个做团购的电商，名字读作‘pīn duō duō’，别搞混了。

**老师原始审核理由**

The target accurately preserves the original meaning, including pronunciation, spelling, and clarification of the name '拼多多' as 'pīn duō duō', with minor grammatical improvement ('是' → '而是') that enhances clarity without altering intent.

**本次核对备注**

这是历史审核“通过”的例子，但仍保留拼字/读音提示；不能当作满足当前精修要求的证明。

### 4. meeting_22（meeting）

**输入（完整原文）**

我这边刚跟智慧园区综合管理平台升级项目团队确认了，完成跨境支付系统与外汇平台接口联调的时间节点是2025年3月13日，这个得确保在客户生命周期价值（CLV）模型优化前完成，不然会影响后续数据同步。

**精修结果（完整原文）**

我已与智慧园区综合管理平台升级项目团队确认，跨境支付系统与外汇平台接口联调的完成时间为2025年3月13日，需确保在客户生命周期价值（CLV）模型优化前完成，以免影响后续数据同步。

**老师原始审核理由**

The target accurately preserves the complete intended meaning, including the confirmed timeline, dependency on CLV model optimization, and the consequence of delay, with minor stylistic improvements and removal of filler words.

### 5. customer_service_27（customer_service）

**输入（完整原文）**

我是在苏宁易购集团买的商品，订单号是2024102901234567，现在想申请退换货，但是关于退货运费的问题有点争议，希望你们能帮忙解决一下。

**精修结果（完整原文）**

我在苏宁易购集团购买了商品，订单号为2024102901234567，现申请退换货，但退货运费存在争议，希望协助解决。

**老师原始审核理由**

目标准确保留了原始意图，包括购买平台、订单号、退换货申请及退货运费争议的诉求，未添加或遗漏关键信息，且已去除冗余表达，符合语义完整性要求。

### 6. academic_23（academic）

**输入（完整原文）**

我们最近在用 EfficientNet-B7 来处理 Open Images V6 上的卫星图像分类任务，结合自监督学习的方法，特别适合做气候变化的长期监测，而且效果比传统的监督模型更稳定，特别是在数据标注有限的情况下。

**精修结果（完整原文）**

我们最近在使用 EfficientNet-B7 处理 Open Images V6 上的卫星图像分类任务，结合自监督学习方法，特别适合气候变化的长期监测，且效果比传统监督模型更稳定，尤其在数据标注有限的情况下。

**老师原始审核理由**

所有关键信息均完整保留，语义无偏差，仅优化了用词和标点，符合原意。

### 7. navigation_12（navigation）

**输入（完整原文）**

我打算从西安坐火车去兰州，走陇海线，对吧？

**精修结果（完整原文）**

我打算从西安坐火车去兰州，走陇海线。

**老师原始审核理由**

去除疑问语气词‘对吧’后，保留了原句的完整意图，未遗漏或添加信息，符合语义边界。

**本次核对备注**

老师把“对吧？”删除判为通过；若问询语气属于用户意图，这类判定仍值得复核。

### 8. dictation_memo_0（dictation_memo）

**输入（完整原文）**

呃，那个，曾强啊，你得赶紧把那个……中国中车集团那边的需求分析给搞一下，对吧，就是那个……需求分析，我跟你说，这个不能拖，之前说要安排团队团建活动，现在资源协调也得跟上，不然大家效率都上不去，对吧，就是说，先把这个需求分析给理清楚，然后团建的事儿也得安排，资源协调也得同步推进，别到时候人都聚不齐，就麻烦了，对吧？

**精修结果（完整原文）**

曾强，你得赶紧把中国中车集团的需求分析搞一下，不能拖。之前说要安排团队团建，现在资源协调也得跟上，不然大家效率都上不去。先理清楚需求分析，再安排团建，资源协调要同步推进，别到时候人都聚不齐，就麻烦了。

**老师原始审核理由**

The target accurately preserves all key points from the input: the urgency of the demand analysis for CRRC, the need for resource coordination, the connection to team building, and the warning about coordination failure. It removes filler and repetition while maintaining the original intent and semantic boundaries. No hallucinated or omitted details are present.

### 9. voice_search_12（voice_search）

**输入（完整原文）**

附近餐厅，搜一下大众点评上的，嗯，不对，是附近餐厅，搜一下大众点评上的，对，就找附近餐厅的，要那个评分高的，还有上一首，别播那个，就附近餐厅，要评分高的。

**精修结果（完整原文）**

附近餐厅，搜一下大众点评上的，要评分高的。

**老师原始审核理由**

The target captures the final intended meaning: searching for nearby restaurants on Dianping with high ratings. The repeated phrases and corrections in the input are consolidated into a clear, final instruction without adding or omitting key details. The mention of '上一首，别播那个' is irrelevant to the core intent and was appropriately ignored.

### 10. vibe_coding_23（vibe_coding）

**输入（完整原文）**

我需要一个函数，用 const 定义一个变量，然后导出它，但不要包含那个错误的语法，比如 SyntaxError: Unexpected token。

**精修结果（完整原文）**

我需要一个函数，用 const 定义一个变量并导出，但不要包含语法错误，例如 SyntaxError: Unexpected token。

**老师原始审核理由**

The target accurately preserves the original intent, removes redundancy ('that' and 'the' are streamlined), and maintains all key details including the avoidance of syntax errors like 'SyntaxError: Unexpected token'. No hallucinated or omitted content is present.

## 历史老师审核拒绝

### 1. voice_search_13（voice_search）

**输入（完整原文）**

暂停播放，等等，我刚才说错了，是智能窗帘，不对，还是暂停播放，对，就是暂停播放。

**精修结果（完整原文）**

暂停播放，我刚才说错了，是智能窗帘，不对，还是暂停播放，对，就是暂停播放。

**老师原始审核理由**

The target omits the repeated '等等' and '还是暂停播放' which are part of the speaker's final intended meaning, indicating hesitation and correction. The original transcript's structure and repetition are essential to capturing the speaker's intent and flow.

**理由中文概述**

老师认为删去“等等”等犹豫、重复内容丢失了原意。

**本次核对备注**

疑似误拒：去掉“等等”符合去口癖目标；目标本身又保留了不少废弃改口，老师给出的具体理由并不可靠。

### 2. navigation_2（navigation）

**输入（完整原文）**

呃……我想问一下，从合肥到南京，走合宁线，对吧？就是……嗯……那个，合肥出发，到南京，应该是走合宁线对吧？哦，等等，不对，不是合宁线，是……好像是二广高速？

**精修结果（完整原文）**

从合肥到南京，走合宁线。

**老师原始审核理由**

The target omits the speaker's correction and final intended meaning, which was that the correct route is not the He-Ning line but rather the Er-Guang Expressway. The final intended meaning is misrepresented by the target.

**理由中文概述**

老师认为目标遗漏最终对“合宁线”的纠正，保留了旧路线。

### 3. navigation_16（navigation）

**输入（完整原文）**

呃……就是，我那个，想从……京开高速，嗯，先走京开高速，然后到深圳北站，对，深圳北站，哎不对，是银百高速，不不，等等，先别急，我重新说，其实我是想走成渝高速，对，成渝高速，然后最后到兰州，对，兰州，就是那个，先到成渝高速，然后再去兰州，对吧

**精修结果（完整原文）**

我想走成渝高速，然后到兰州，最后回到京开高速。

**老师原始审核理由**

The target incorrectly reverses the route sequence and adds 'back to Jingkai Expressway' which was not mentioned in the final intended meaning. The speaker's final intended route is solely from Chengyu Expressway to Lanzhou, not involving Jingkai Expressway at the end.

**理由中文概述**

老师认为目标新增了“最后回到京开高速”，原文最终路线没有这个要求。

### 4. navigation_11（navigation）

**输入（完整原文）**

呃，那个，我我我先想查一下从广州到桂林，就是广桂线那条路线，嗯，对，就是广桂线，对吧？就是广州到桂林的广桂线，啊，不对，等一下，我刚刚好像说错了，我其实想问的是苏州，对，苏州那边的路线，嗯，就是苏州，对，苏州，对，就是苏州，就是那个苏州，啊，对，就是苏州，对吧？

**精修结果（完整原文）**

我想查询从广州到桂林的广桂线路线，不对，我实际上想问的是苏州的路线，就是苏州的路线。

**老师原始审核理由**

The target omits the speaker's repeated confirmation and hesitation markers ('呃', '那个', '对吧', '啊', '对', '就是', '对，对，就是') that are part of the original intent and flow. It also fails to preserve the speaker's self-correction process and the final emphasis on '苏州' as the intended destination, reducing the nuance of the speaker's uncertainty and eventual clarification. The final intended meaning includes the speaker's verbal hesitation and iterative confirmation, which are essential to the complete meaning.

**理由中文概述**

老师认为删除口癖、犹豫和反复确认损失了原意。

**本次核对备注**

审核理由与当前去口癖要求冲突：犹豫词和反复确认不应因为被删除而直接拒绝。目标本身也尚未收敛到最终意图。

### 5. daily_chat_22（daily_chat）

**输入（完整原文）**

我前两天去武汉玩，登了黄鹤楼，五层楼高51米，爬上去大概花了半小时，风景特别好，而且现在用华为的FreeBuds Pro 3听歌，降噪效果真不错，还有空间音频，感觉像在演唱会现场，顺带还带了大疆Mini 4 Pro拍了些视频，避障功能挺靠谱的，4K HDR拍得也清楚。

**精修结果（完整原文）**

我前两天去武汉玩，登了黄鹤楼，五层楼高51米，爬上去大约花了半小时，风景特别好。现在用华为FreeBuds Pro 3听歌，降噪效果不错，还有空间音频，感觉像在演唱会现场。

**老师原始审核理由**

Target omits details about using DJI Mini 4 Pro to shoot videos, its obstacle avoidance function, and 4K HDR quality, which were part of the speaker's complete intended meaning.

**理由中文概述**

老师认为目标漏掉了大疆 Mini 4 Pro、避障和 4K HDR 录像的信息。

### 6. dictation_memo_17（dictation_memo）

**输入（完整原文）**

呃，那个，我刚刚在想啊，就是关于那个季度营销计划，嗯，要重新做一下，对吧？就是说，MacBook Pro 14英寸，这个设备得先确认一下，嗯，对，就是用这个设备来处理，然后呢，预算的话，大概是六千万，六千万，就是六千万人民币，对，六千万元，然后项目规划这块儿，得重新更新项目进度表，对，要更新那个进度表，但等一下，刚才我说的是更新进度表，其实我本意是想说——

**精修结果（完整原文）**

我刚刚在想，关于季度营销计划需要重新制定。MacBook Pro 14英寸设备需先确认，用于处理相关工作。预算约为六千万元人民币。项目规划方面需先制定季度营销计划，再更新项目进度表。

**老师原始审核理由**

target omits the speaker's correction intent and misrepresents the sequence of actions; original transcript indicates the speaker intended to correct 'update progress table' to 'develop marketing plan first', but target incorrectly frames 'develop marketing plan' as the initial step, contradicting the speaker's final intended sequence

**理由中文概述**

老师认为目标误解了末尾改口和行动顺序。原始输入在改口途中就截断了。

**本次核对备注**

输入未说完，不应根据未出现的后文推断最终改口内容；请结合完整记录审看。

### 7. voice_search_14（voice_search）

**输入（完整原文）**

呃，那个，我想听一下……嗯……QQ音乐排行榜，就是那个……嗯……最近最火的歌，对，就是那个……呃，还有，能不能顺便给我查一下知乎日报，看看有没有什么新的文章推荐，啊，不对，等等，我好像记错了，其实是想查一下汽车维修店的评价，对，就是最近那个……那个……花店，对，花店的推荐，然后最后再……嗯……分享照片，就是……那个……刚才拍的那张，对，就是那个……对，还是先听QQ音乐排行榜，对，QQ音乐排行榜，就是那个……对，就是这样。

**精修结果（完整原文）**

我想听一下QQ音乐排行榜，最近最火的歌。顺便查一下知乎日报的新文章推荐。不对，等等，我记错了，其实是想查汽车维修店的评价。对，就是最近那个花店的推荐。

**老师原始审核理由**

Target omits the final intent to share the photo and fails to capture the complete sequence of corrected intentions, including the final clarification about the photo. The final intended meaning includes sharing a photo, which is not reflected in the target.

**理由中文概述**

老师认为目标漏掉了分享照片等后续意图。

### 8. customer_service_20（customer_service）

**输入（完整原文）**

就是我那个……那个订单，就是2024年10月12号下的，那个单号是2024101234567890，买的是小米14 Ultra，结果现在收到货了，发现金额对不上，好像多扣了……我是在贵州省贵阳市南明区解放路128号那边收的货，本来应该付六千八，结果账单上显示七千二，这差了四百块，我这有点搞不清楚，是不是vivo通信科技有限公司那边系统出问题了？

**精修结果（完整原文）**

我于2024年10月12日下单，订单号为2024101234567890，购买小米14 Ultra，收货地址为贵州省贵阳市南明区解放路128号，实际应付款6800元，账单显示7200元，相差400元。怀疑vivo通信科技有限公司系统存在错误。

**老师原始审核理由**

Target incorrectly attributes the issue to 'vivo communication technology company' which was not mentioned in the input; the speaker did not specify the company responsible, and this addition constitutes a hallucinated detail.

**理由中文概述**

老师称输入没有提到 vivo 公司，而目标无依据地添加了公司归因。

**本次核对备注**

明显存在理由错误：输入末尾确实写着“是不是vivo通信科技有限公司那边系统出问题了？”。

### 9. dictation_memo_7（dictation_memo）

**输入（完整原文）**

嗯……那个，王芳，王芳对吧？她那个请假申请，嗯，审批员工请假申请，对，我得赶紧处理一下，呃，其实黄磊那边也……等等，我刚想说黄磊，但其实是安排外部审计工作，对，安排外部审计工作，那个……嗯，先放一放，还是先处理王芳的请假申请，是的，王芳的，没错，然后那个……智能手表 Apple Watch Series 9，我得确认一下，是不是要给财务部的同事配，啊，不对，不是这个，先别管手表，先处理请假的事，王芳的，对，必须先审批，对，王芳的请假申请，最终还是得先处理这个。

**精修结果（完整原文）**

王芳的请假申请需要尽快审批。黄磊那边的工作是安排外部审计，可以暂时放一放。先处理王芳的请假申请，确认是否为财务部同事配备Apple Watch Series 9，但当前应优先处理请假事宜。

**老师原始审核理由**

The target introduces unsupported details (e.g., 'confirming if Apple Watch Series 9 is to be provided to a finance department colleague') not present in the input. The input never confirms or discusses assigning watches to anyone. The final intended meaning is solely about prioritizing the approval of Wang Fang's leave application, with the watch mention being a distraction the speaker explicitly rejects ('not this, don't worry about the watch'). The target also misrepresents the speaker's intent by framing the watch as a pending decision, which contradicts the speaker's clear directive to 'not worry about it' and focus only on the leave. The target thus hallucinates a decision point not in the input.

**理由中文概述**

老师认为目标保留了已经被说话者撤回的手表事项。

### 10. vibe_coding_54（vibe_coding）

**输入（完整原文）**

呃，就是这个……我那个 Flask 项目里头，想加个过滤器，对，就是那个 filter，对吧？然后呢，之前用 PHP 写过类似的逻辑，但那个太乱了，现在想换成 Grails，但我不太熟 Grails 的语法，就卡在这儿了……哦对，那个 throw 异常的地方，好像也得改一下，不然会出问题，啊不是，应该是……等等，其实我真正想说的是，最后得用 Flask 的 filter，不是 Grails，我刚脑子一抽，差点就往 Grails 上写了，哎，搞错了，应该是 Flask 的 filter，不是 Grails 的，对吧？

**精修结果（完整原文）**

我那个 Flask 项目里想加一个过滤器，之前用 PHP 写过类似逻辑，但太混乱了，现在想换成 Grails，但不太熟悉 Grails 语法，卡在这儿了。不过其实我真正想说的是，最后得用 Flask 的 filter，不是 Grails 的，刚才脑子一抽差点写错，应该是 Flask 的 filter，不是 Grails 的。

**老师原始审核理由**

The target incorrectly retains the mention of switching to Grails, which the speaker explicitly corrected as a mistake. The final intended meaning is that the speaker wants to use Flask's filter, not Grails, and the target fails to fully reflect this correction by including the erroneous Grails reference.

**理由中文概述**

老师认为目标保留了已经被撤回的“换成 Grails”的旧意图。
