---
annotations_creators:
- expert-generated
language_creators:
- crowdsourced
language:
- en
- zh
license:
- cc-by-sa-4.0
multilinguality:
- multilingual
size_categories:
- 10K<n<100K
source_datasets:
- original
task_categories:
- automatic-speech-recognition
task_ids: []
pretty_name: 'ASCEND: A Spontaneous Chinese-English Dataset for Code-switching in
  Multi-turn Conversation'
tags:
- speech-recognition
- code-switching
dataset_info:
  config_name: main
  features:
  - name: id
    dtype: string
  - name: path
    dtype: string
  - name: audio
    dtype:
      audio:
        sampling_rate: 16000
  - name: transcription
    dtype: string
  - name: duration
    dtype: float32
  - name: language
    dtype: string
  - name: original_speaker_id
    dtype: int64
  - name: session_id
    dtype: int64
  - name: topic
    dtype: string
  splits:
  - name: train
    num_bytes: 1014573740.14
    num_examples: 9869
  - name: test
    num_bytes: 106171230.135
    num_examples: 1315
  - name: validation
    num_bytes: 106772517.43
    num_examples: 1130
  download_size: 1223536062
  dataset_size: 1227517487.7050002
configs:
- config_name: main
  data_files:
  - split: train
    path: main/train-*
  - split: test
    path: main/test-*
  - split: validation
    path: main/validation-*
  default: true
---

# Dataset Card for ASCEND

## Table of Contents
- [Dataset Description](#dataset-description)
  - [Dataset Summary](#dataset-summary)
  - [Supported Tasks](#supported-tasks-and-leaderboards)
  - [Languages](#languages)
- [Usage](#usage)
- [Dataset Structure](#dataset-structure)
  - [Data Splits](#data-instances)
- [Additional Information](#additional-information)
  - [Licensing Information](#licensing-information)
  - [Citation Information](#citation-information)

## Dataset Description

- **Homepage:** [Needs More Information]
- **Repository:** [Needs More Information]
- **Paper:** https://arxiv.org/abs/2112.06223
- **Leaderboard:** [Needs More Information]
- **Point of Contact:** [Needs More Information]

### Dataset Summary

ASCEND (A Spontaneous Chinese-English Dataset) introduces a high-quality resource of spontaneous multi-turn conversational dialogue Chinese-English code-switching corpus collected in Hong Kong. ASCEND consists of 10.62 hours of spontaneous speech with a total of ~12.3K utterances. The corpus is split into 3 sets: training, validation, and test with a ratio of 8:1:1 while maintaining a balanced gender proportion on each set.

### Supported Tasks and Leaderboards

Code-switching

### Languages

Chinese and English

## Usage

To obtain the full dataset (complete with train, validation, and test set), simply run this:

```
import datasets
dataset = datasets.load_dataset("CAiRE/ASCEND")
```

## Dataset Structure

A typical data point comprises the path to the audio file, the loaded audio array, and its transcription. Additional fields include datapoint id, duration, language, speaker id, session id, and topic.

```
{
	'id': '00644',
	'path': '.cache/huggingface/datasets/downloads/extracted/f0b33b5266cd9452ee310eef3577cf7adb7f29aa54dbff74b9a8ee406a55d614/waves/ses2_spk3_L13101_189.900_5.490.wav',
	'audio': {
		'path': '.cache/huggingface/datasets/downloads/extracted/f0b33b5266cd9452ee310eef3577cf7adb7f29aa54dbff74b9a8ee406a55d614/waves/ses2_spk3_L13101_189.900_5.490.wav',
		'array': array([-6.1035156e-05, -1.8310547e-04, 3.0517578e-05, ...,
			0.0000000e+00, -3.0517578e-05, 0.0000000e+00
		], dtype = float32),
		'sampling_rate': 16000
	},
	'transcription': '因为你不可能邀你的female friends去说走我们去play basketball',
	'duration': 5.489999771118164,
	'language': 'mixed',
	'original_speaker_id': 3,
	'session_id': 2,
	'topic': 'sports'
}
```

### Data Splits

Number of utterances: 9,869 train, 1,130 validation, and 1,315 test.

## Additional Information

For comprehensive explanations, please check [our paper](https://arxiv.org/pdf/2112.06223.pdf).

### Licensing Information

Creative Common Attribution Share-Alike 4.0 International (CC-BY-SA 4.0)

### Citation Information

If you use our dataset, please cite us:

```
@inproceedings{lovenia2022ascend,
  title={ASCEND: A Spontaneous Chinese-English Dataset for Code-switching in Multi-turn Conversation},
  author={Lovenia, Holy and Cahyawijaya, Samuel and Winata, Genta Indra and Xu, Peng and Yan, Xu and Liu, Zihan and Frieske, Rita and Yu, Tiezheng and Dai, Wenliang and Barezi, Elham J and others},
  booktitle={Proceedings of the 13th Language Resources and Evaluation Conference (LREC)},
  year={2022}
 ```