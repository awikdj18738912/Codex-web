"""Check exact frozen counts and Qwen assistant loss masks before full SFT."""
import argparse,hashlib,json,os
from pathlib import Path
import yaml

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',type=Path,required=True);p.add_argument('--report',type=Path,required=True);a=p.parse_args()
    os.environ.update(CUDA_VISIBLE_DEVICES='',HF_HUB_OFFLINE='1',TRANSFORMERS_OFFLINE='1',TOKENIZERS_PARALLELISM='false')
    import torch
    torch.set_num_threads(8)
    from llamafactory.data import get_dataset,get_template_and_fix_tokenizer
    from llamafactory.hparams import get_train_args
    from llamafactory.model import load_tokenizer
    config=yaml.safe_load(a.config.read_text()); data=Path(config['dataset_dir'])
    m=json.loads((data/'manifest.json').read_text())
    for name,expected in m['files_sha256'].items():
        assert hashlib.sha256((data/name).read_bytes()).hexdigest()==expected,name
    assert config['finetuning_type'] in {'full','lora'} and config['model_name_or_path']=='/home/aim0/data/ASR/models/Qwen2.5-0.5B-Instruct'
    if config['finetuning_type']=='lora':
        assert config['lora_rank']==16 and config['lora_alpha']==32 and config['lora_target']=='all'
    assert config['train_on_prompt'] is False and config['packing'] is False
    cpu=dict(config,use_cpu=True,bf16=False,fp16=False,output_dir=str(a.report.parent/'argument_check'),dataloader_num_workers=0,preprocessing_num_workers=4)
    model_args,data_args,train_args,_,_=get_train_args(cpu)
    module=load_tokenizer(model_args);tokenizer=module['tokenizer'];template=get_template_and_fix_tokenizer(tokenizer,data_args)
    ds=get_dataset(template,model_args,data_args,train_args,stage='sft',**module)
    result={'status':'passed','dataset_counts':{},'examples_checked':0,'max_tokens':0,'assistant_only_loss':True,'candidate_references_not_human_gold':True,'config_sha256':hashlib.sha256(a.config.read_bytes()).hexdigest(),'dataset_manifest_sha256':hashlib.sha256((data/'manifest.json').read_bytes()).hexdigest()}
    for key,split in [('train_dataset','train'),('eval_dataset','dev')]:
        dataset=ds[key];assert len(dataset)==m['split_counts'][split],split
        result['dataset_counts'][split]=len(dataset)
        for batch in dataset.iter(batch_size=256):
            for ids,labels in zip(batch['input_ids'],batch['labels']):
                start=next((i for i,t in enumerate(labels) if t!=-100),None)
                assert start is not None and start>0 and labels[start:]==ids[start:],'loss_mask'
                assert len(ids)<config['cutoff_len'],'possible_truncation'
                result['max_tokens']=max(result['max_tokens'],len(ids));result['examples_checked']+=1
    a.report.write_text(json.dumps(result,ensure_ascii=False,indent=2)+'\n');print(json.dumps(result,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
