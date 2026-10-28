# v4.3 提示词修改及配对对照

状态：提示词和对照脚本已准备，尚未进行教师模型生成，不能声称质量提升。

依据：v4.2 中自我改口多数退化成填充或重复；断句指令明确要求“短语内部”，导致拆词；中英混用类别允许原样复制。

修改：允许被明确撤回的局部错误表达；定义改口与停顿的差别；断句放在完整词之间的依存关系处；中英混用要求中文侧有实际口语现象；重复与未完成起头从同时要求改为二选一。保留数字单位等现有程序校验，不把校验改善计为提示词效果。

## 配对运行

启动原有 GPU0 qwen38-teacher 服务（8012）后，依次运行：

```bash
python results/qwen38_prompt_v4_3_controlled_20261022/generate_comparison.py --variant baseline --output results/qwen38_prompt_v4_3_controlled_20261022/baseline.jsonl --concurrency 8 --retries 0
python results/qwen38_prompt_v4_3_controlled_20261022/generate_comparison.py --variant revised --output results/qwen38_prompt_v4_3_controlled_20261022/revised.jsonl --concurrency 8 --retries 0
```

当前已调整为两张GPU各启动一个教师副本（8012/8013），每卡最大8序列，两组客户端各8并发，总请求并发16；同一配对case按哈希固定落在同一张卡。推荐使用 `python results/qwen38_prompt_v4_3_controlled_20261022/run_comparison.py` 同时调度两组。旧单卡部分输出已归档到 archived_single_gpu_partial，未计入新对照。

两组使用完全相同的100个种子、类别、随机种子规则、temperature=0.65、top_p=0.8、max_tokens=768、校验和重试次数。baseline 精确保留 v4.2 的系统提示词和类别指令；它不是论文原版提示词，对论文原版的比较需要另外适配原始输入契约。

盲审应混排两组，隐藏版本与指定类别，先评源文本是否自然、最终意图是否忠实、实际出现何种口语现象，再解盲匹配要求。以全部请求为分母报告严格有效对产出率，分别记录拒答、原样复制、语义错误、类别不匹配。程序直接复制的12条 passthrough 仅用于管线控制，不能计入教师生成成功或提示词提升。不得用这批已用于调参的种子充当独立最终测试集。
