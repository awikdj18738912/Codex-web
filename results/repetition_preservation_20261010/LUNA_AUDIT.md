# Luna独立只读审计

主代理根据gpt-6-luna回复整理，子代理未修改源码或运行GPU。

- 删除入口包括window精修、拒绝后的safe salvage、模型提出删除的二审、正常local review及确定性清理。共用校验覆盖提交路径，仅过滤候选不足以覆盖这些入口。
- 新`repetition_intent_loss`不能被只豁免`semantic_content_loss`的旧local例外绕过；数字重复递归路径也保留理由。
- 最初的全文重复计数存在“另一处相同串替代本处”的漏洞，主代理改为具体编辑位置校验，新增回归；cue扫描改用预计算句界与二分位置查询。
- Web preflight在缓存REMOVE及神经复核之前执行；拒绝窗口后二审也先检查。提交层仍再次校验。
- replay中的student先在临时候选finder中评分，再进入Web preflight，故Refiner调用数不等于整个门控成本。
- 24条是text-only开发者行为预期，没有音频gold。6条rules基线也因保护改变，不能把全部最终匹配提升归因于模型/门控。
- 最终guard版current匹配22/24、student17/24；各自相对rules改善10/5、退化均0；Refiner25/7、plausibility各3。
- student仍有5条错跳开发预期；报告应保留其失败，不能只报保护效果。
- 提示消融输出与guard一致，current多1次调用，不能宣称在其他分布上通用无效；默认实现不采用提示改动。
- 86条operation proposals仅train来源，5行由旧教师的uncertain投影局部benefit，实际4条录音；原标签未变、没有自动训练，未核验音频，不能当新教师/实际音频金标。
- 单字否定、无意图线索的数字复述、后续窗口才出现的cue等是明确保留策略边界；不能称为学到语义意图或全覆盖。
- 最终22条测试与14条风险开发回放摘要与逐条记录一致；087保留全文，062/008仍完成重复清理，008的姓名残余差异保留。
- Luna指出教师依赖白名单仍可遗漏transcript_repairs、repetition_plausibility、quantifiers、Web辅助函数或采集器；主代理最终加入全部冻结system目录文件及采集适配器哈希，案例/输入数据不参与教师等价比较。
