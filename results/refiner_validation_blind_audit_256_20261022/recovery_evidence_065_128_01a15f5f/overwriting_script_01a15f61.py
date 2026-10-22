import json,copy,collections
from pathlib import Path
p=Path('/home/aim0/桌面/tan/ASR/Codex-web/results/refiner_validation_blind_audit_256_20261022')
cases=[json.loads(l) for l in (p/'cases_065_128.jsonl').read_text().splitlines()]
old={x['case_no']:x for x in map(json.loads,(p/'review_065_128.jsonl').read_text().splitlines())}
rows={}
for c in cases:
 n=c['case_no'];rows[n]={'case_no':n,'id':c['id'],'source_status':'CLEAR','reference':{'verdict':'PASS','issues':[],'reason':'保留输入事实、最终意图及有意义的解释，未见实质错误。','proposed_reference':None},**{k:{'meaning':'PASS','cleanup':'PASS','fidelity':'PASS','overall':'PASS','issues':[],'reason':'事实、最终意图与英文标识保持；现有口语表达不构成明显清理失败。'} for k in ['A','B']},'winner':'TIE','winner_reason':'两者均保留有效信息，无实质质量差距。'}
def issue(tag,q,d):return {'tag':tag,'quote':q,'detail':d}
def out(n,k,dim,v,q,d):
 x=rows[n][k];x[dim]=v;x['issues'].append(issue(dim,q,d));x['overall']='FAIL' if 'FAIL' in [x[t] for t in ['meaning','cleanup','fidelity']] else ('UNJUDGEABLE' if 'UNJUDGEABLE' in [x[t] for t in ['meaning','fidelity']] else ('MINOR' if 'MINOR' in [x[t] for t in ['meaning','cleanup','fidelity']] else 'PASS'));x['reason']='；'.join(t['detail'] for t in x['issues'])
def ref(n,v,q,d,pro=None):rows[n]['reference']={'verdict':v,'issues':[issue('cleanup' if '改口' in d or '重复' in d else 'fidelity',q,d)],'reason':d,'proposed_reference':pro}
def win(n,w,r):rows[n]['winner']=w;rows[n]['winner_reason']=r
# 明显未清理的大段结巴/改口，证据逐条选取。
clean_fail={65:('A','这不……这不……这不就是……'),67:('AB','不是2010万'),70:('B','不是新源那拉提草原，应该是新源县'),73:('B','呃，就是说，像那个TUM RGB-D那个benchmark，对吧'),74:('A','这个地址，这个地址，是……是……是发件地吧'),78:('A','因为……都是数据传输和实时性的问题对吧？然后……嗯……'),79:('AB','是延边朝鲜族自治州，不是新昌县'),81:('B','那个叫1024的……不是，是1024的那集'),87:('A','呃那个啊，其实我刚刚在想啊，就是说咱们那个医院手术室那边，对吧'),88:('AB','——不对，还是改回疏附站'),92:('B','呃那个就是我这个……我之前买的那个'),96:('B','这个……其实上个月的数据有点……嗯……不太稳定'),97:('AB','起点是广州南站，终点是深圳北站'),103:('AB','最终，还是得先恢复备份'),105:('AB','但实际应该是1.5米，不过等等，我再核对一下，其实还是1.2米'),108:('B','就是，就是……60，对，就是60那个档'),109:('A','啊，但其实我刚才说的这个是贝叶斯，不是量子机器学习，我绕了……重新来'),115:('A','驻马店西站，就是京广线那个站'),117:('A','600，600，就是600，那个……600'),118:('B','那个，那个，那个，那个'),121:('A','彭宇恒是……不是，是张伟'),123:('B','其实更合理的应该是，应该是……对，应该调低一点'),128:('A','这个事，我得……得……得，还得重新说一遍')}
for n,(ks,q) in clean_fail.items():
 for k in ks:out(n,k,'cleanup','FAIL',q,'保留明显即时改口链、大段结巴或反复重述，未完成必要清理。')
# 轻微口癖不以FAIL惩罚；技术解释与真实历史修正予以保留。
minor_clean={65:('B','我这不是……这不是要我多跑一趟嘛'),66:('AB','不是，是去参加他那个深夜读书会'),71:('A','呃，就是我刚才在跑那个Fortran的代码'),77:('AB','哎你别听别人瞎说啊，那个啥'),83:('A','呃，就是我这边那个飞利浦Hue智能灯泡套装'),84:('B','哎你说那个……'),98:('B','呃，就是说，我这边在搞那个 Webhook API 的时候'),99:('B','呃，那个，就是，我想问一下'),102:('AB','嗯……就是那个感觉'),106:('A','这些地方，都得导航到，对，都得导航到'),112:('B','呃，这个是……2024年10月29号'),115:('B','不是驻马店，我刚才说反了，是郴州西站'),124:('B','那个……我要去塔城站，嗯'),126:('A','呃，那个，就是，我想查一下')}
for n,(ks,q) in minor_clean.items():
 for k in ks:out(n,k,'cleanup','MINOR',q,'少量口癖、停顿或短小重复仍可清理，但不属于大段清理失败。')
# 参考稿改口链使用与A/B一致标准。
ref(67,'FAIL','4150万，不是2010万，是4150万','保留即时金额改口及结尾整段重复确认，应只留最终金额与任务信息。','预算为4150万，用于支持中国科学院武汉岩土力学研究所的项目。另外，需要组织跨部门技术研讨会，时间为3月20号，由林知微牵头负责。')
ref(79,'FAIL','是延边朝鲜族自治州，不是新昌县','保留新昌县、延边、新昌县的即时改口链及重复，应提取最终目的地。','我想查一下去新昌县的路线。先给我导航到新昌县，我再看看有没有别的路。')
ref(88,'FAIL','等等，改成去泉州东站（福厦线）','未删除撤回的地点候选，保留完整即时改口链。','从疏附到疏附站（南疆线）。')
ref(97,'FAIL','起点是广州南站，终点是深圳北站','未清理被明确撤回的起终站及重复改口。','我要从广州坐广深城际线去深圳，从广州站到深圳站，中途停靠东莞站和虎门站。')
ref(103,'FAIL','最终，还是得先恢复备份','保留多轮重复确认和即时改口，且留有未完成的“然后再……”片段。','谭雅文你记得吧，就是中国农业科学院植物保护研究所生物安全与外来物种防控研究室的。她之前说的恢复备份是第一要务，得先恢复备份，再处理OAuth2令牌刷新。')
ref(105,'FAIL','但实际应该是1.5米，不过等等，我再核对一下，其实还是1.2米','即时数值改口链完整照录，未清理已撤回的1.5米候选。','关于2024年12月1日的项目验收，我再核对后确认定位误差是1.2米。这个误差值会影响地下管廊巡检机器人的漏检率，得优先处理。')
ref(66,'PASS','是去参加他举办的深夜读书会','“他那个深夜读书会”允许自然整理为“他举办的”，没有足够证据认定新增实质事实。')
rows[66]['reference']['issues']=[]
ref(74,'MINOR','但那个是发件地','将“是发件地吧”及“好像没人接”确定化；延迟疑问也改为明确断言。', '我下单时选了“极速发货”，本以为第2天能收到，结果快2天还没收到，有点急。我查物流，显示2024年11月29号才发货，这不就是延迟超过48小时了吗？我买的是宜家PAX衣柜系统，200厘米高，带滑动门和内置照明，本想赶紧装好，现在真闹心。我的地址是吉林省长春市南关区人民大街100号东北师范大学本部。之前联系客服好像没人接，希望你们尽快处理、发货，不然房子都快装不下去了。还有江苏省苏州市工业园区星湖街123号，我朋友也想买这个，但这个地址是发件地吧？那不是我的地址，我是长春那个地址，别再拖了，行不行？')
ref(78,'MINOR','这与我们之前讨论的AI问卷隐私合规性问题有关','将“好像有点关联”略作确定化，宜保留推测语气。', '我刚看到研究组的报告，提到企业远程会议中音视频延迟和讨论效率下降的主观评分对比。我觉得这与之前讨论的AI问卷隐私合规性问题好像有点关联，因为都涉及数据传输和实时性。虽然时间上不太对，但2024-07-29这个日期是不是也暗示我们得尽快评估这些系统在实际使用中的影响？还有城市公共厕所智能除臭系统，虽然看起来不相关，但调控逻辑也依赖实时数据反馈。我们是不是该把这类系统的隐私和延迟问题统一纳入跨部门会议的讨论框架？')
out(78,'B','fidelity','MINOR','这个日期也暗示了我们得尽快评估','将询问日期是否暗示行动的推测改成肯定陈述。')
out(75,'A','fidelity','FAIL','我前两天在厦门曾厝垵的海风图书馆','遗漏明确定位信息“海边民宿街第7号”，不能以数值书写格式差异解释。')
ref(86,'MINOR','戴森空气净化器','删除同一品牌的英文名称Dyson；中文品牌还在，未错换实体，但未完整保留英文名称。',next(c['reference'] for c in cases if c['case_no']==86).replace('戴森空气净化器','戴森Dyson空气净化器'))
ref(87,'MINOR','经常无法稳定采样','把“有时候”强化为“经常”，并去掉系统“好像”出问题的保留语气。',next(c['reference'] for c in cases if c['case_no']==87).replace('最近出了点问题','最近好像出了点问题').replace('经常无法稳定采样','有时候根本无法稳定采样'))
ref(92,'MINOR','是否也需要一并处理','遗漏询问是否还需提供其他信息这一附带请求。',next(c['reference'] for c in cases if c['case_no']==92).replace('另外，我还想问一下','我是不是还得再提供别的信息？另外，我还想问一下'))
out(92,'A','fidelity','FAIL','就是我之前填的收货地址','把仍不确定的收货地址确认成事实。')
# 未完成句或同字纠正须保持歧义，不补写。
for n in [81,93,104,117,120,122]:rows[n]['source_status']='AMBIGUOUS'
for n in [81,104,117,120,122]:
 rows[n]['reference']={'verdict':'UNJUDGEABLE','issues':[],'reason':{81:'1024的同字改口未解决，不能安全删除或猜测节目定位。',104:'“不是翡翠的翠，是翠绿的翠”同字纠正未解决。',117:'跨主题指令中的600缺少含义，无法确定完整执行意图。',120:'豆腐手艺、降噪、分贝和视频分辨率无意义拼接，不能推断完整关系。',122:'“不是丹霞或丹夏”存在同字否定矛盾，无法确定本来要排除的字。'}[n],'proposed_reference':None}
 for k in ['A','B']:
  rows[n][k]['meaning']='UNJUDGEABLE';rows[n][k]['fidelity']='UNJUDGEABLE'
  rows[n][k]['overall']='FAIL' if rows[n][k]['cleanup']=='FAIL' else 'UNJUDGEABLE'
  rows[n][k]['reason']='源文本存在未解决歧义；可独立观察的清理问题另列。'
# 122 B删除同字否定属于合理清理，但仍无法验证原本排除对象。
ref(93,'FAIL','整个山城的夜景都尽收眼底','补写输入未完成句的谓语；“就让人忍不住”同样是无依据补全。')
out(93,'A','fidelity','FAIL','整个山城的夜景都尽收眼底','输入在“夜景都……”处中断，不能自行补写“尽收眼底”。')
out(93,'B','fidelity','FAIL','才5.6秒的短视频刷起来倒快','删除5000mAh容量并将“才”与5.6秒拼接，改变容量相关事实。')
ref(96,'MINOR','需要再核对一下','把“可能得再核对”提升为明确需要，轻度改变建议的确定性。',next(c['reference'] for c in cases if c['case_no']==96).replace('需要再核对一下','可能需要再核对一下').replace('需优化','可能需要优化'))
out(96,'A','fidelity','MINOR','目前运行正常','将“运行得还行”略强化为正常运行。')
out(98,'A','meaning','FAIL','检查了Go的并发处理和Rails的middleware有没有bug','把正在考虑排查的可能原因写成已完成检查，改变动作状态。')
out(90,'A','meaning','FAIL','下月1号北京的天气预报：晴','答复查询结果，未做原请求的文字精修。')
out(90,'A','fidelity','FAIL','气温10℃-25℃','凭空新增温度、风况及空气质量，并把滤芯提醒改成使用空气净化器提醒。')
out(103,'B','meaning','FAIL','然后再恢复备份','将停顿“然后再……”拼成再次恢复备份的顺序步骤，新增重复操作。')
out(109,'B','fidelity','FAIL','适用于实时任务','输入只说降低计算量、能跑动，没有实时任务结论。')
ref(109,'FAIL','再通过sigmoid激活完成非线性映射','省略了可独立成立的贝叶斯后验公式及先验、似然、归一化解释，还新增“完成非线性映射”。','量子机器学习做分子性质预测挺火，可以用变分量子算法拟合分子能级或预测电子结构性质。贝叶斯公式P(θ|D) ∝ P(D|θ)P(θ)描述的是后验推断，需要先验与似然相乘后归一化，不能直接混为量子机器学习。可以用轻量化Transformer结构Swin-Lite处理分子图数据，相较计算量大的传统Swin，稀疏注意力能节省算力。sigmoid激活函数σ(z) = 1 / (1 + e^-z)也用于神经网络的MLP层，不能说它是量子的。核心是量子机器学习结合轻量化Transformer结构，用贝叶斯先验建模参数不确定性，最后用sigmoid激活，避免术语混淆。')
out(111,'A','fidelity','MINOR','孩子们玩得投入','普通英文词kids恰当译为中文，依英文保留要求最多轻微问题。')
ref(111,'MINOR','孩子们玩得非常投入','普通英文词kids被恰当译为中文，按英文保留要求至多轻微问题。',next(c['reference'] for c in cases if c['case_no']==111).replace('孩子们玩得非常投入','kids玩得非常投入'))
out(118,'A','meaning','FAIL','后来我们尝试将Q-learning与经验回放缓冲区结合','把“想，能不能结合”的设想写成已尝试，改变实验状态。')
out(121,'B','fidelity','FAIL','彭宇恒负责相关工作','重新引入已撤回的负责人，并漏掉单细胞RNA流程优化和先部署再核对的要求。')
ref(126,'MINOR','啊对，是星期五。','去掉“对吧”使待确认星期五变为确定事实。',next(c['reference'] for c in cases if c['case_no']==126).replace('啊对，是星期五。','是星期五，对吧？'))
out(126,'B','fidelity','MINOR','是星期五。','把“星期五，对吧”的待确认语气改为断言。')
out(127,'A','fidelity','MINOR','能缓解这个问题','把“好像能缓解”提升为确定能缓解。')
out(127,'A','cleanup','FAIL','等等，我刚说错了，R是整体准确率，但其实是不对，应该叫accuracy','仍保留指标命名的即时改口链，且把犹疑断句连成“整体准确率不对”的错误歧义。')
out(127,'B','cleanup','FAIL','等等，我刚说错了，R是整体准确率，但其实是……不对，应该叫accuracy','大段结巴及指标命名即时改口未清理；条件概率误解的历史解释应保留。')
ref(128,'FAIL','收票单位为杭州阿里巴巴网络技术有限公司','输入只称其为地址，新增收票单位身份关系。',next(c['reference'] for c in cases if c['case_no']==128).replace('收票单位为杭州阿里巴巴网络技术有限公司','地址是杭州阿里巴巴网络技术有限公司'))
out(73,'A','fidelity','MINOR','结果也没问题，反而比传统方法强','将“好像也没”的保留语气略作确定化；主要实验结论未变。')
ref(73,'MINOR','但实际表现反而优于传统方法','将“结果好像也没”略作确定化，宜保留不确定性。',next(c['reference'] for c in cases if c['case_no']==73).replace('但实际表现反而优于传统方法','但结果好像没有出问题，反而比传统方法强'))
# 逐条比较最终评分：保真优先，其次清理。无实质差异不因标点变化定胜负。
for n in rows:
 a,b=rows[n]['A'],rows[n]['B'];rank={'PASS':0,'MINOR':1,'UNJUDGEABLE':2,'FAIL':3}
 if a['overall']==b['overall']:continue
 if rank[a['overall']]<rank[b['overall']]:win(n,'A','A的有效信息保留与清理质量更好，B存在上列问题。')
 else:win(n,'B','B的有效信息保留与清理质量更好，A存在上列问题。')
for n in [67,79,88,97,105]:win(n,'TIE','两者均未清理即时改口链；最终事实相同不抵消清理失败。')
win(65,'B','B显著减少结巴与重复，仅有少量残留；A清理失败。')
win(73,'A','A清理了大段口癖并保留评估方法解释；Chamfer距离为合理译法，不构成名称错换。')
win(92,'B','B虽清理失败，但保留地址不确定性；A把待确认地址写成事实，保真问题更重。')
win(93,'TIE','A补写未完成事实，B丢失容量并错误拼接；两者都有独立实质错误。')
win(103,'A','两者清理均失败，B还把断句拼成再次恢复备份的新增操作，A保真较好。')
win(109,'A','两者不可直接使用，但A保留事实；B新增实时任务结论，保真更差。')
win(121,'A','A清理失败；B除漏信息还重新引入撤回的负责人，保真更差。')
win(127,'TIE','两者均保留改口链；A虽较简洁又削弱不确定性，综合不足以判定胜者。')
for n in [104,120,122]:win(n,'UNJUDGEABLE','源文本歧义尚未解决，两者无可独立认定的实质新增或丢失，不能据此判定胜负。')
win(81,'A','两者节目编号意图均不能安全判定；A清理明显更充分，B保留大段结巴。')
win(117,'B','600的含义仍待人工确认；B保留该值并去掉大段重复，A清理失败。')
# 每项结论使用对应文本的逐字证据；保存审核变更细节。
logs=[]
for c in cases:
 n=c['case_no'];r=rows[n]
 for k,field in [('reference','reference'),('A','output_A'),('B','output_B')]:
  for i in r[k]['issues']:
   assert i['quote'] in c[field],(n,k,i)
 for field in ['source_status','reference','A','B','winner','winner_reason']:
  if r[field]!=old[n][field]:logs.append({'case_no':n,'field':field,'old':old[n][field],'new':r[field],'reason': r[field]['reason'] if isinstance(r[field],dict) else (r['winner_reason'] if field.startswith('winner') else '按未解决同字纠正、未完成句或无意义跨主题拼接标源歧义。')})
(p/'adjudicated_065_128.jsonl').write_text(''.join(json.dumps(rows[n],ensure_ascii=False)+'\n' for n in sorted(rows)))
(p/'adjudication_log_065_128.jsonl').write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in logs))
print('rows',len(rows),'log entries',len(logs))
for field in ['source_status','reference','A','B','winner']:
 print(field,dict(collections.Counter((r[field].get('verdict',r[field].get('overall')) if isinstance(r[field],dict) else r[field]) for r in rows.values())))
