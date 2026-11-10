#!/usr/bin/env python3
"""
Generate a valid BF16 safetensors MTP draft-layer shard for Qwen 3.6 35B A3B.
Format strictly matches MTPAttachTool specification in MferenceRepack.
"""

import sys
import os
import json
import struct

def main():
    out_path = sys.argv[1] if len(sys.argv) > 1 else "scratch/qwen36_mtp_shard.safetensors"
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    
    # Qwen 3.6 dimensions
    d = 2048
    f = 512
    num_heads = 16
    num_kv_heads = 2
    head_dim = 256
    q_dim = num_heads * head_dim   # 4096
    kv_dim = num_kv_heads * head_dim # 512
    
    tensors_spec = [
        ("mtp.fc.weight", [d, 2 * d]),
        ("mtp.pre_fc_norm_embedding.weight", [d]),
        ("mtp.pre_fc_norm_hidden.weight", [d]),
        ("mtp.norm.weight", [d]),
        ("mtp.layers.0.input_layernorm.weight", [d]),
        ("mtp.layers.0.post_attention_layernorm.weight", [d]),
        ("mtp.layers.0.self_attn.q_proj.weight", [2 * q_dim, d]),
        ("mtp.layers.0.self_attn.k_proj.weight", [kv_dim, d]),
        ("mtp.layers.0.self_attn.v_proj.weight", [kv_dim, d]),
        ("mtp.layers.0.self_attn.o_proj.weight", [d, q_dim]),
        ("mtp.layers.0.self_attn.q_norm.weight", [head_dim]),
        ("mtp.layers.0.self_attn.k_norm.weight", [head_dim]),
        ("mtp.layers.0.mlp.gate_proj.weight", [f, d]),
        ("mtp.layers.0.mlp.up_proj.weight", [f, d]),
        ("mtp.layers.0.mlp.down_proj.weight", [d, f]),
    ]
    
    header = {}
    current_offset = 0
    tensor_buffers = []
    
    for name, shape in tensors_spec:
        elem_count = 1
        for s in shape:
            elem_count *= s
        byte_length = elem_count * 2 # BF16 = 2 bytes
        
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [current_offset, current_offset + byte_length]
        }
        current_offset += byte_length
        # Initialize with zeros (norms will become 1.0 via addOne, weights are neutral)
        tensor_buffers.append(b'\x00' * byte_length)
        
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_len = len(header_json)
    
    print(f"Writing Safetensors MTP Shard: {out_path}")
    print(f"  Tensors:       {len(tensors_spec)}")
    print(f"  Header Length: {header_len} bytes")
    print(f"  Payload Size:  {current_offset / (1024*1024):.2f} MB")
    
    with open(out_path, "wb") as f_out:
        f_out.write(struct.pack("<Q", header_len))
        f_out.write(header_json)
        for buf in tensor_buffers:
            f_out.write(buf)
            
    print(f"✅ Generated valid shard: {out_path} ({os.path.getsize(out_path)} bytes)")

if __name__ == "__main__":
    main()
