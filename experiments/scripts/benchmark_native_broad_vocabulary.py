#!/usr/bin/env python3
"""Run the existing paired native benchmark with the broad 2023 vocabulary model."""
from __future__ import annotations
import argparse
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(Path(__file__).resolve().parent))
import benchmark_native_entity_streaming as n

MODEL_DIR=Path("/home/aim0/data/models/ASR/sherpa-onnx-streaming-zipformer-small-bilingual-zh-en-2023-02-16")

def make_recognizer(condition:str,score:float):
    import sherpa_onnx
    common=dict(tokens=str(MODEL_DIR/"tokens.txt"),
        encoder=str(MODEL_DIR/"encoder-epoch-99-avg-1.int8.onnx"),
        decoder=str(MODEL_DIR/"decoder-epoch-99-avg-1.onnx"),
        joiner=str(MODEL_DIR/"joiner-epoch-99-avg-1.int8.onnx"),
        num_threads=2,enable_endpoint_detection=False,modeling_unit="cjkchar",provider="cpu")
    if condition=="greedy":common["decoding_method"]="greedy_search"
    else:common.update(decoding_method="modified_beam_search",max_active_paths=4)
    common["hotwords_score"]=score
    return sherpa_onnx.OnlineRecognizer.from_transducer(**common)

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--run-dir",type=Path,required=True,help="base entity_backend_comparison run directory")
    p.add_argument("--split",choices=("dev","test","control","all"),default="dev")
    p.add_argument("--conditions",nargs="+",choices=("greedy","beam","hotword"),default=("beam","hotword"))
    p.add_argument("--scores",nargs="+",type=float,default=(2.0,))
    p.add_argument("--limit",type=int)
    p.add_argument("--num-shards",type=int,default=1)
    p.add_argument("--shard-index",type=int,default=0)
    args=p.parse_args()
    if args.num_shards<1 or not 0<=args.shard_index<args.num_shards:p.error("invalid shard configuration")
    base=args.run_dir.resolve();target=base/"broad_vocab";target.mkdir(parents=True,exist_ok=True)
    upstream=base/"inputs.jsonl";linked=target/"inputs.jsonl"
    if not upstream.is_file():raise FileNotFoundError(upstream)
    if linked.is_symlink():
        if linked.resolve()!=upstream.resolve():raise ValueError(f"unexpected inputs symlink target: {linked.resolve()}")
    elif linked.exists():
        raise ValueError(f"refusing to replace non-symlink {linked}")
    else:linked.symlink_to(upstream)
    n.MODEL_DIR=MODEL_DIR
    n.make_recognizer=make_recognizer
    args.run_dir=target
    n.run(args)

if __name__=="__main__":main()
