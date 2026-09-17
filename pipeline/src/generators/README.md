# 文本生成器

生成器负责构造训练样本中的语义内容和口语/书面形式：

- seed_generator.py：按场景生成并去重种子池。
- oral_text_generator.py：将种子扩展为带场景特征和口语现象的 Oral 转写。
- clean_text_generator.py：将 Oral 文本改写为意图保持的 Clean 目标。
- asr_simulator.py：通过 LLM、文本规则或 TTS/ASR 路径生成对齐的 ASR 风格输入。
- __init__.py：包导出。

Oral 表示接近说话方式的输入；Clean 表示期望的精修目标。修改提示词时要同时核对二者的内容一致性。
