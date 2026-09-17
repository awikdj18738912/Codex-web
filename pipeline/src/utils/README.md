# 流水线通用工具

- io_utils.py：JSON/JSONL 读写、日志与可续跑检查点。
- prompt_loader.py：加载并渲染配置化提示词。
- keyword_utils.py：提取、规范化和检查生成样本中的关键词/审核目标。
- __init__.py：包标记。

生成阶段会依赖这些函数保持记录 schema、日志和续跑行为一致。改动序列化逻辑时应检查旧数据兼容性和中断续跑结果。
