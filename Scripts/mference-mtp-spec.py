#!/usr/bin/env python3
"""MTP (Multi-Token Prediction) Spec & Generator Template for Qwen 3.6 35B.

Defines the exact tensor names, dimensions, and safetensors contract required
to attach an MTP speculative draft layer to Qwen 3.6 in Mference.
"""
import json
import os
import sys

MANIFEST_PATH = "/Users/vladslugin/Documents/Mference/scratch/qwen36.gturbo/manifest.json"

def main():
    if not os.path.exists(MANIFEST_PATH):
        print(f"Error: manifest not found at {MANIFEST_PATH}")
        sys.exit(1)

    with open(MANIFEST_PATH, "r") as f:
        manifest = json.load(f)

    arch = manifest["arch"]
    d = arch["hiddenSize"]          # 2048
    f = arch.get("moeIntermediateSize") or arch.get("ffnIntermediate", 512) # 512
    num_heads = arch["numHeads"]     # 16
    num_kv_heads = arch.get("numFullKVHeads") or arch.get("numKVHeads", 2) # 2
    head_dim = arch.get("fullHeadDim") or arch.get("headDim", 256) # 256

    q_dim = num_heads * head_dim     # 4096 (or 2 * q_dim = 8192 if gated)
    kv_dim = num_kv_heads * head_dim # 512

    print("=" * 68)
    print("  QWEN 3.6 MTP DRAFT LAYER ARCHITECTURAL CONTRACT")
    print("=" * 68)
    print(f"Hidden Size (d):         {d}")
    print(f"FFN Intermediate (f):    {f}")
    print(f"Q Heads x Dim:           {num_heads} x {head_dim} ({q_dim})")
    print(f"KV Heads x Dim:          {num_kv_heads} x {head_dim} ({kv_dim})")
    print("-" * 68)

    tensors = [
        {"name": "mtp.fc.weight", "shape": [d, 2 * d], "desc": "Projects [norm_embed(tok), norm_hidden(h)] -> hidden"},
        {"name": "mtp.pre_fc_norm_embedding.weight", "shape": [d], "desc": "Zero-centered RMSNorm for embedding input"},
        {"name": "mtp.pre_fc_norm_hidden.weight", "shape": [d], "desc": "Zero-centered RMSNorm for trunk hidden input"},
        {"name": "mtp.norm.weight", "shape": [d], "desc": "Final RMSNorm before lm_head"},
        {"name": "mtp.layers.0.input_layernorm.weight", "shape": [d], "desc": "MTP Attention input layernorm"},
        {"name": "mtp.layers.0.post_attention_layernorm.weight", "shape": [d], "desc": "MTP MLP input layernorm"},
        {"name": "mtp.layers.0.self_attn.q_proj.weight", "shape": [2 * q_dim, d], "desc": "Gated Query projection [query; gate]"},
        {"name": "mtp.layers.0.self_attn.k_proj.weight", "shape": [kv_dim, d], "desc": "Key projection"},
        {"name": "mtp.layers.0.self_attn.v_proj.weight", "shape": [kv_dim, d], "desc": "Value projection"},
        {"name": "mtp.layers.0.self_attn.o_proj.weight", "shape": [d, q_dim], "desc": "Output projection"},
        {"name": "mtp.layers.0.self_attn.q_norm.weight", "shape": [head_dim], "desc": "Per-head Query norm"},
        {"name": "mtp.layers.0.self_attn.k_norm.weight", "shape": [head_dim], "desc": "Per-head Key norm"},
        {"name": "mtp.layers.0.mlp.gate_proj.weight", "shape": [f, d], "desc": "MTP SwiGLU gate projection"},
        {"name": "mtp.layers.0.mlp.up_proj.weight", "shape": [f, d], "desc": "MTP SwiGLU up projection"},
        {"name": "mtp.layers.0.mlp.down_proj.weight", "shape": [d, f], "desc": "MTP SwiGLU down projection"},
    ]

    total_bytes = 0
    for t in tensors:
        elements = 1
        for s in t["shape"]:
            elements *= s
        b = elements * 2 # BF16 = 2 bytes
        total_bytes += b
        print(f"  {t['name']:<42} {str(t['shape']):<15} ({b / 1024 / 1024:.2f} MB)")

    print("-" * 68)
    print(f"Total MTP Shard Raw Size: {total_bytes / 1024 / 1024:.2f} MB")
    print(f"INT4 Quantized Target:   ~{total_bytes / 4 / 1024 / 1024:.2f} MB (only ~18 MB in RAM!)")
    print("=" * 68)

    # Save to JSON spec
    out_spec = "/Users/vladslugin/Documents/Mference/scratch/qwen36_mtp_spec.json"
    with open(out_spec, "w") as f:
        json.dump({"model": "qwen36", "tensors": tensors, "total_raw_bytes": total_bytes}, f, indent=2)
    print(f"Saved spec to: {out_spec}")

if __name__ == "__main__":
    main()
