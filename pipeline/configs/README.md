# 数据流水线配置

配置模块位于 pipeline/configs/，负责场景定义、生成提示词和流水线运行默认值。

- settings.py：服务 URL、环境变量、场景比例、数据路径及生成参数。
- scenes.py：场景注册表，以及语言、噪声和场景元数据。
- prompts.json：种子文本生成提示词。
- prompts_oral.json：Oral 口语转写生成提示词。
- prompts_clean.json：从 Oral 生成 Clean 书面目标的提示词。
- __init__.py：Python 包标记。

调整提示词或场景比例后，建议在小批次上检查输入输出 schema 和口语/书面目标差异，再运行完整生成流程。模型服务配置通过环境变量提供；不要把密钥写入仓库。
