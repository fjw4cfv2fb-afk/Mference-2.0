#!/usr/bin/env python3
"""
Harvest Layer 39 weights from Qwen 3.6 to construct an intelligent MTP draft shard.
- Dequantizes Layer 39 attention (q, k, v, o, q_norm, k_norm) and shared expert MLP (gate, up, down).
- Copies final norm and layer norms.
- Constructs mtp.fc with an identity hidden bypass [2048, 4096] (W[:, 2048+i] = 1.0).
- Sets pre_fc norms to 0.0 (converted to 1.0 identity via addOne in MTPAttachTool).
Output is a valid BF16 Safetensors shard ready for MTPAttachTool.
"""

import sys
import os
import struct
import json

def bf16_to_float(bits: int) -> float:
    return struct.unpack(">f", struct.pack(">HH", bits, 0))[0]

def float_to_bf16(val: float) -> int:
    bits = struct.unpack(">I", struct.pack(">f", val))[0]
    lsb = (bits >> 16) & 1
    rounding_bias = 0x7FFF + lsb
    return ((bits + rounding_bias) >> 16) & 0xFFFF

def dequant_int4_affine_row(packed_bytes: bytes, scale_words: list, bias_words: list, cols: int) -> list:
    groups = cols // 64
    out_bf16 = [0] * cols
    
    for g in range(groups):
        s = bf16_to_float(scale_words[g])
        b = bf16_to_float(bias_words[g])
        g_bytes = packed_bytes[g * 32 : (g + 1) * 32]
        
        for k in range(32):
            byte = g_bytes[k]
            lo = byte & 0x0F
            hi = (byte >> 4) & 0x0F
            v_even = lo * s + b
            v_odd = hi * s + b
            out_bf16[g * 64 + k * 2] = float_to_bf16(v_even)
            out_bf16[g * 64 + k * 2 + 1] = float_to_bf16(v_odd)
            
    return out_bf16

def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo_root, "scratch/qwen36.gturbo")
    out_shard = sys.argv[2] if len(sys.argv) > 2 else os.path.join(repo_root, "scratch/qwen36_layer39_harvested_mtp.safetensors")
    
    weights_path = os.path.join(model_dir, "model_weights.bin")
    if not os.path.exists(weights_path):
        print(f"Error: {weights_path} not found")
        sys.exit(1)
        
    print(f"Reading resident index from {weights_path}...")
    with open(weights_path, "rb") as f:
        hdr = f.read(24)
        index_size, resident_size, entry_count = struct.unpack("<QQQ", hdr)
        region = f.read(index_size - 24)
        
        entries = {}
        for i in range(entry_count):
            p = i * 72
            name_off, name_len, dtype = struct.unpack("<IHB", region[p:p+7])
            name = region[name_off - 24 : name_off - 24 + name_len].decode("utf-8")
            foff, sz, s0, s1, s2, s3, sc_off, sc_sz, b_off, b_sz = struct.unpack("<QQIIIIQQQQ", region[p+8:p+72])
            entries[name] = {
                "dtype": dtype, "offset": foff, "size": sz,
                "shape": (s0, s1), "sc_offset": sc_off, "sc_size": sc_sz,
                "b_offset": b_off, "b_size": b_sz
            }
            
        print(f"Found {len(entries)} resident tensors. Harvesting Layer 39 weights...")
        
        # Helper to read raw bytes from weights file
        def read_tensor_bytes(e):
            f.seek(e["offset"])
            return f.read(e["size"])
            
        def dequant_tensor(name, rows, cols):
            e = entries[name]
            f.seek(e["offset"])
            packed = f.read(e["size"])
            f.seek(e["sc_offset"])
            scale_bytes = f.read(e["sc_size"])
            f.seek(e["b_offset"])
            bias_bytes = f.read(e["b_size"])
            
            groups_per_row = cols // 64
            scales = struct.unpack(f"<{len(scale_bytes)//2}H", scale_bytes)
            biases = struct.unpack(f"<{len(bias_bytes)//2}H", bias_bytes)
            
            bytes_per_row = cols // 2
            out_words = []
            for r in range(rows):
                r_packed = packed[r * bytes_per_row : (r + 1) * bytes_per_row]
                r_scales = scales[r * groups_per_row : (r + 1) * groups_per_row]
                r_biases = biases[r * groups_per_row : (r + 1) * groups_per_row]
                out_words.extend(dequant_int4_affine_row(r_packed, r_scales, r_biases, cols))
            return struct.pack(f"<{len(out_words)}H", *out_words)
            
        def read_bf16_norm(name, count, subtract_one=False):
            e = entries[name]
            f.seek(e["offset"])
            raw = f.read(e["size"])
            words = list(struct.unpack(f"<{count}H", raw))
            if subtract_one:
                # w_shard = w_model - 1.0 (so MTPAttachTool w_shard + 1.0 = w_model)
                adjusted = []
                for w in words:
                    v = bf16_to_float(w) - 1.0
                    adjusted.append(float_to_bf16(v))
                return struct.pack(f"<{count}H", *adjusted)
            return raw

        tensors = {}
        
        # 1. Attention INT4 -> BF16
        print("  Dequantizing QKV + O projections...")
        tensors["mtp.layers.0.self_attn.q_proj.weight"] = (
            [8192, 2048],
            dequant_tensor("language_model.model.layers.39.self_attn.q_proj.weight", 8192, 2048)
        )
        tensors["mtp.layers.0.self_attn.k_proj.weight"] = (
            [512, 2048],
            dequant_tensor("language_model.model.layers.39.self_attn.k_proj.weight", 512, 2048)
        )
        tensors["mtp.layers.0.self_attn.v_proj.weight"] = (
            [512, 2048],
            dequant_tensor("language_model.model.layers.39.self_attn.v_proj.weight", 512, 2048)
        )
        tensors["mtp.layers.0.self_attn.o_proj.weight"] = (
            [2048, 4096],
            dequant_tensor("language_model.model.layers.39.self_attn.o_proj.weight", 2048, 4096)
        )
        
        # 2. Shared Expert MLP INT4 -> BF16
        print("  Dequantizing Shared Expert MLP...")
        tensors["mtp.layers.0.mlp.gate_proj.weight"] = (
            [512, 2048],
            dequant_tensor("language_model.model.layers.39.mlp.shared_expert.gate_proj.weight", 512, 2048)
        )
        tensors["mtp.layers.0.mlp.up_proj.weight"] = (
            [512, 2048],
            dequant_tensor("language_model.model.layers.39.mlp.shared_expert.up_proj.weight", 512, 2048)
        )
        tensors["mtp.layers.0.mlp.down_proj.weight"] = (
            [2048, 512],
            dequant_tensor("language_model.model.layers.39.mlp.shared_expert.down_proj.weight", 2048, 512)
        )
        
        # 3. Norms (with subtract_one=True so MTPAttachTool addOne reconstructs exact weights)
        print("  Harvesting layernorms and head norms...")
        tensors["mtp.layers.0.input_layernorm.weight"] = (
            [2048],
            read_bf16_norm("language_model.model.layers.39.input_layernorm.weight", 2048, subtract_one=True)
        )
        tensors["mtp.layers.0.post_attention_layernorm.weight"] = (
            [2048],
            read_bf16_norm("language_model.model.layers.39.post_attention_layernorm.weight", 2048, subtract_one=True)
        )
        tensors["mtp.layers.0.self_attn.q_norm.weight"] = (
            [256],
            read_bf16_norm("language_model.model.layers.39.self_attn.q_norm.weight", 256, subtract_one=True)
        )
        tensors["mtp.layers.0.self_attn.k_norm.weight"] = (
            [256],
            read_bf16_norm("language_model.model.layers.39.self_attn.k_norm.weight", 256, subtract_one=True)
        )
        tensors["mtp.norm.weight"] = (
            [2048],
            read_bf16_norm("language_model.model.norm.weight", 2048, subtract_one=True)
        )
        
        # 4. Identity pre_fc norms (0.0 in shard -> +1.0 in MTPAttachTool = 1.0 identity)
        tensors["mtp.pre_fc_norm_embedding.weight"] = ([2048], b'\x00' * (2048 * 2))
        tensors["mtp.pre_fc_norm_hidden.weight"] = ([2048], b'\x00' * (2048 * 2))
        
        # 5. FC projection [2048, 4096]: Identity bypass on hidden (W[:, 2048+i] = 1.0)
        print("  Constructing mtp.fc with Identity Hidden Bypass [2048, 4096]...")
        d = 2048
        fc_words = [0] * (d * 2 * d)
        bf16_one = float_to_bf16(1.0) # 0x3F80
        for i in range(d):
            # Row i, Column (d + i) -> bypass hidden directly
            fc_words[i * (2 * d) + (d + i)] = bf16_one
        tensors["mtp.fc.weight"] = ([d, 2 * d], struct.pack(f"<{len(fc_words)}H", *fc_words))

    # Assemble Safetensors file
    print(f"Packaging {len(tensors)} harvested tensors into Safetensors...")
    header = {}
    current_offset = 0
    payloads = []
    
    for name, (shape, data) in tensors.items():
        byte_len = len(data)
        header[name] = {
            "dtype": "BF16",
            "shape": shape,
            "data_offsets": [current_offset, current_offset + byte_len]
        }
        current_offset += byte_len
        payloads.append(data)
        
    header_json = json.dumps(header, separators=(',', ':')).encode('utf-8')
    header_len = len(header_json)
    
    with open(out_shard, "wb") as f_out:
        f_out.write(struct.pack("<Q", header_len))
        f_out.write(header_json)
        for p in payloads:
            f_out.write(p)
            
    print(f"✅ Successfully harvested Layer-39 MTP Shard: {out_shard} ({os.path.getsize(out_shard):,} bytes)")

if __name__ == "__main__":
    main()
