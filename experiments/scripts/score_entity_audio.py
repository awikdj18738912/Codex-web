"""Isolated acoustic evidence for bounded glossary candidates.

No hints or reference are supplied to ASR. Compare the likelihood of the
authoritative source and each one-span replacement on identical PCM. Weights
are read-only; this process does not alter the running ASR/Refiner service.
"""
from __future__ import annotations
import argparse
import hashlib
import json
import math
from pathlib import Path
import time


class AudioLikelihoodVerifier:
    def __init__(self,model_path,device='cuda:0',use_prefix_cache=True):
        import torch
        from qwen_asr import Qwen3ASRModel
        self.torch=torch;torch.set_num_threads(4)
        self.asr=Qwen3ASRModel.from_pretrained(model_path,dtype=torch.bfloat16,device_map=device,
                    attn_implementation='sdpa',max_inference_batch_size=1,max_new_tokens=256)
        self.asr.model.eval();self.thinker=self.asr.model.thinker
        self.tokenizer=self.asr.processor.tokenizer
        self.use_prefix_cache=use_prefix_cache

    def compare(self,pcm,source,candidates):
        import numpy as np
        torch=self.torch;wave=np.frombuffer(pcm,dtype='<i2').astype(np.float32)/32768
        prompt=self.asr._build_text_prompt(context='',force_language='Chinese')
        with torch.inference_mode():
            features=self.asr.processor(text=[prompt],audio=[wave],padding=True,return_tensors='pt')
            features=features.to(self.asr.device).to(self.asr.dtype)
            prefix_ids=features['input_ids'];prefix_len=prefix_ids.shape[1]
            prefix=self.thinker.get_input_embeddings()(prefix_ids)
            audio=self.thinker.get_audio_features(features['input_features'],feature_attention_mask=features['feature_attention_mask'])
            mask=self.thinker.get_placeholder_mask(prefix_ids,prefix)
            prefix=prefix.masked_scatter(mask,audio.to(prefix.dtype))
            cache=None;first_logits=None
            if self.use_prefix_cache:
                attention=torch.ones(prefix.shape[:2],dtype=torch.long,device=self.asr.device)
                positions,_=self.thinker.get_rope_index(attention)
                encoded=self.thinker.model(inputs_embeds=prefix,attention_mask=attention,
                                position_ids=positions,use_cache=True)
                cache=encoded.past_key_values
                assert cache.get_seq_length()==prefix_len
                first_logits=self.thinker.lm_head(encoded.last_hidden_state[:,-1:]).float()
            def likelihood(text):
                ids=self.tokenizer(text,add_special_tokens=False,return_tensors='pt')['input_ids'].to(self.asr.device)
                if not ids.numel():raise ValueError('Transcript is empty')
                inputs=torch.cat([prefix,self.thinker.get_input_embeddings()(ids)],dim=1)
                attention=torch.ones(inputs.shape[:2],dtype=torch.long,device=self.asr.device)
                positions,_=self.thinker.get_rope_index(attention)
                if cache is None:
                    hidden=self.thinker.model(inputs_embeds=inputs,attention_mask=attention,
                                    position_ids=positions,use_cache=False).last_hidden_state
                    logits=self.thinker.lm_head(hidden[:,prefix_len-1:-1]).float()
                else:
                    assert cache.get_seq_length()==prefix_len
                    try:
                        hidden=self.thinker.model(inputs_embeds=inputs[:,prefix_len:],attention_mask=attention,
                                    position_ids=positions[:,:,prefix_len:],past_key_values=cache,use_cache=True,
                                    cache_position=torch.arange(prefix_len,inputs.shape[1],device=self.asr.device)).last_hidden_state
                        logits=torch.cat([first_logits,self.thinker.lm_head(hidden[:,:-1]).float()],dim=1)
                    finally:
                        cache.crop(prefix_len)
                    assert cache.get_seq_length()==prefix_len
                logp=torch.log_softmax(logits,dim=-1).gather(-1,ids.unsqueeze(-1)).squeeze(-1)
                return dict(sum=float(logp.sum()),mean=float(logp.mean()),tokens=ids.numel())
            def likelihood_batch(texts):
                if cache is None:return [likelihood(text) for text in texts]
                sequences=[self.tokenizer.encode(text,add_special_tokens=False) for text in texts]
                lengths=torch.tensor([len(ids) for ids in sequences],device=self.asr.device)
                ids=torch.nn.utils.rnn.pad_sequence([torch.tensor(seq,device=self.asr.device) for seq in sequences],
                             batch_first=True,padding_value=self.tokenizer.pad_token_id)
                body_mask=torch.arange(ids.shape[1],device=self.asr.device)[None,:]<lengths[:,None]
                attention=torch.cat([torch.ones(len(texts),prefix_len,device=self.asr.device,dtype=torch.long),body_mask.long()],dim=1)
                positions,_=self.thinker.get_rope_index(attention)
                assert cache.get_seq_length()==prefix_len
                cache.batch_repeat_interleave(len(texts))
                try:
                    hidden=self.thinker.model(inputs_embeds=self.thinker.get_input_embeddings()(ids),attention_mask=attention,
                        position_ids=positions[:,:,prefix_len:],past_key_values=cache,use_cache=True,
                        cache_position=torch.arange(prefix_len,prefix_len+ids.shape[1],device=self.asr.device)).last_hidden_state
                    logits=torch.cat([first_logits.expand(len(texts),-1,-1),self.thinker.lm_head(hidden[:,:-1]).float()],dim=1)
                    logp=torch.log_softmax(logits,dim=-1).gather(-1,ids.unsqueeze(-1)).squeeze(-1)
                    sums=(logp*body_mask).sum(-1).cpu().tolist();sizes=lengths.cpu().tolist()
                    return [dict(sum=total,mean=total/size,tokens=size) for total,size in zip(sums,sizes)]
                finally:
                    cache.crop(prefix_len)
                    cache.batch_select_indices(torch.tensor([0],device=self.asr.device))
                    assert cache.get_seq_length()==prefix_len
            baseline=likelihood(source);scores=[];proposals=[]
            for candidate in candidates:
                a,b=candidate['start'],candidate['end']
                if not 0<=a<b<=len(source) or source[a:b]!=candidate['observed']:raise ValueError('Stale candidate span')
                proposals.append(source[:a]+candidate['canonical']+source[b:])
            for offset in range(0,len(proposals),3):
                group=likelihood_batch([source]+proposals[offset:offset+3]);paired_baseline=group[0]
                for candidate,result in zip(candidates[offset:offset+3],group[1:]):
                    # Evaluate the source in the SAME batch to reduce effects
                    # of numerical differences between decoder kernel shapes.
                    delta=result['sum']-paired_baseline['sum']
                    scores.append(dict(candidate=candidate,likelihood=result,paired_baseline_likelihood=paired_baseline,
                                       log_likelihood_delta=delta,score=1/(1+math.exp(-max(-60,min(60,delta))))))
        return dict(baseline_likelihood=baseline,candidates=scores,prefix_cache=self.use_prefix_cache,candidate_batch_size=3,
                    model_input='Identical original PCM + source/candidate transcript, empty ASR context')


def main():
    p=argparse.ArgumentParser();p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--model',default='/home/aim0/data/models/ASR/Qwen3-ASR-1.7B');p.add_argument('--no-prefix-cache',action='store_true');a=p.parse_args()
    rows=json.loads(a.manifest.read_text())['samples'];a.output.parent.mkdir(parents=True,exist_ok=True)
    done={(r['id'],r['variant']) for r in [json.loads(l) for l in a.output.read_text().splitlines()] } if a.output.exists() else set()
    verifier=AudioLikelihoodVerifier(a.model,use_prefix_cache=not a.no_prefix_cache)
    for i,row in enumerate(rows):
        if (row['id'],row['variant']) in done:continue
        pcm=Path(row['pcm_path']).read_bytes()
        if hashlib.sha256(pcm).hexdigest()!=row['pcm_sha256']:raise ValueError('PCM changed')
        started=time.perf_counter();result=verifier.compare(pcm,row['source'],row['candidates'])
        record=dict(id=row['id'],variant=row['variant'],source=row['source'],pcm_sha256=row['pcm_sha256'],
                    elapsed_seconds=time.perf_counter()-started,**result)
        with a.output.open('a') as stream:stream.write(json.dumps(record,ensure_ascii=False)+'\n')
        print(i+1,'/',len(rows),row['id'],row['variant'],round(record['elapsed_seconds'],2),
              [(r['candidate']['observed'],r['candidate']['canonical'],round(r['log_likelihood_delta'],2)) for r in result['candidates']],flush=True)


if __name__=='__main__':main()
