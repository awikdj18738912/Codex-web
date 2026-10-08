"""Entity-specific pair verifier. It scores bounded choices and cannot rewrite.

Loading is explicit; existing entities do not acquire a new dependency or
behavior merely because this optional module exists.
"""
from __future__ import annotations
import json
import math
from pathlib import Path

FEATURE_NAMES=('phonetic_score','character_score','length_score','retrieval_score',
               'same_first_character','observed_length','canonical_length','span_boundary_ok')


def candidate_input(text,candidate):
    start,end=candidate['start'],candidate['end'];observed=candidate['observed'];canonical=candidate['canonical']
    if not 0<=start<end<=len(text) or text[start:end]!=observed:
        raise ValueError('Candidate offsets must refer to the authoritative source')
    left=text[max(0,start-40):start];right=text[end:end+40]
    original=left+'[unused1]'+observed+'[unused2]'+right
    proposed=left+'[unused1]'+canonical+'[unused2]'+right
    description={'PERSON':'姓名','ORG':'机构名称','LOC':'地点','TERM':'专业术语'}.get(candidate['entity_type'],candidate['entity_type'])
    pair=description+'：'+canonical+'。规范后：'+proposed
    features=[candidate[k] for k in FEATURE_NAMES[:4]]
    features += [float(observed[0]==canonical[0]),min(24,len(observed))/24,min(24,len(canonical))/24,float(not candidate['boundary_issue'])]
    return original,pair,features


def build_network(encoder_path):
    import torch
    from transformers import AutoModel
    class PairVerifier(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.encoder=AutoModel.from_pretrained(encoder_path)
            self.head=torch.nn.Sequential(torch.nn.Linear(self.encoder.config.hidden_size+len(FEATURE_NAMES),128),
                                          torch.nn.GELU(),torch.nn.Dropout(0.1),torch.nn.Linear(128,1))
        def forward(self,input_ids,attention_mask,features,token_type_ids=None):
            kwargs=dict(input_ids=input_ids,attention_mask=attention_mask)
            if token_type_ids is not None:kwargs['token_type_ids']=token_type_ids
            encoded=self.encoder(**kwargs).last_hidden_state[:,0]
            return self.head(torch.cat([encoded,features.to(encoded.dtype)],dim=-1)).squeeze(-1)
    return PairVerifier()


class EntityPairRanker:
    def __init__(self,path: str|Path,device='cpu'):
        import torch
        from transformers import AutoTokenizer
        self.torch=torch;self.device=device;self.path=Path(path)
        self.metadata=json.loads((self.path/'metadata.json').read_text())
        if self.metadata['feature_names']!=list(FEATURE_NAMES):raise ValueError('Unsupported entity feature schema')
        self.tokenizer=AutoTokenizer.from_pretrained(self.path/'encoder')
        self.model=build_network(self.path/'encoder')
        self.model.load_state_dict(torch.load(self.path/'verifier.pt',map_location='cpu',weights_only=True))
        self.model.to(device).eval()

    def scores(self,text,candidates,batch_size=64):
        output=[]
        with self.torch.inference_mode():
            for offset in range(0,len(candidates),batch_size):
                values=[candidate_input(text,c) for c in candidates[offset:offset+batch_size]]
                encoded=self.tokenizer([v[0] for v in values],[v[1] for v in values],padding=True,truncation=True,max_length=160,return_tensors='pt')
                encoded={k:v.to(self.device) for k,v in encoded.items()}
                features=self.torch.tensor([v[2] for v in values],dtype=self.torch.float32,device=self.device)
                output.extend(self.torch.sigmoid(self.model(**encoded,features=features)).float().cpu().tolist())
        return output


def apply_verified_choices(text,candidates,scores,*,threshold,minimum_margin=0.25):
    """Only copy registered candidate spellings at their original positions."""
    if len(candidates)!=len(scores):raise ValueError('One score per candidate is required')
    if not math.isfinite(threshold) or not 0<=threshold<=1:raise ValueError('Invalid acceptance threshold')
    if not math.isfinite(minimum_margin) or not 0<=minimum_margin<=1:raise ValueError('Invalid acceptance margin')
    if any(not math.isfinite(s) or not 0<=s<=1 for s in scores):raise ValueError('Invalid verifier score')
    selected=[]
    ranked=sorted(zip(candidates,scores),key=lambda cs:(-cs[1],cs[0]['start'],cs[0]['end'],cs[0]['entity_id']))
    for candidate,score in ranked:
        if score<threshold:continue
        start,end=candidate['start'],candidate['end']
        if not 0<=start<end<=len(text) or text[start:end]!=candidate['observed']:continue
        if any(start<c['end'] and c['start']<end for c,_ in selected):continue
        rivals=[s for c,s in ranked if c['entity_id']!=candidate['entity_id'] and start<c['end'] and c['start']<end]
        if rivals and score-max(rivals)<minimum_margin:continue
        selected.append((candidate,score))
    output=text
    for candidate,score in sorted(selected,key=lambda cs:cs[0]['start'],reverse=True):
        output=output[:candidate['start']]+candidate['canonical']+output[candidate['end']:]
    return output,[dict(**c,verifier_score=s) for c,s in selected]
