---
library_name: peft
license: other
base_model: /home/aim0/data/models/ASR/AgenticASR-Refiner
tags:
- base_model:adapter:/home/aim0/data/models/ASR/AgenticASR-Refiner
- llama-factory
- lora
- transformers
pipeline_tag: text-generation
model-index:
- name: refiner_lora_20261010
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# refiner_lora_20261010

This model is a fine-tuned version of [/home/aim0/data/models/ASR/AgenticASR-Refiner](https://huggingface.co//home/aim0/data/models/ASR/AgenticASR-Refiner) on the refiner_local_train dataset.
It achieves the following results on the evaluation set:
- Loss: 0.1625

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 0.0001
- train_batch_size: 1
- eval_batch_size: 1
- seed: 20261010
- gradient_accumulation_steps: 8
- total_train_batch_size: 8
- optimizer: Use OptimizerNames.ADAMW_TORCH with betas=(0.9,0.999) and epsilon=1e-08 and optimizer_args=No additional optimizer arguments
- lr_scheduler_type: cosine
- lr_scheduler_warmup_steps: 0.05
- num_epochs: 3

### Training results

| Training Loss | Epoch | Step | Validation Loss |
|:-------------:|:-----:|:----:|:---------------:|
| 0.1191        | 1.0   | 79   | 0.1704          |
| 0.0247        | 2.0   | 158  | 0.1750          |
| 0.0196        | 3.0   | 237  | 0.1625          |


### Framework versions

- PEFT 0.18.1
- Transformers 5.8.0
- Pytorch 2.11.0+cu130
- Datasets 4.0.0
- Tokenizers 0.22.2