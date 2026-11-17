#!/usr/bin/env python3
"""Generate an architectural tensor dossier of the active Mference model.

Produces a complete manifest of tensor names, shapes, quantization schemes,
and byte layouts to guide weight-space programming and steering adjustments.
"""
import json
import os
import struct
import sys

HDR = 24
ENT = 72


def parse_tensor_index(bin_path):
    if not os.path.exists(bin_path):
        return []
    with open(bin_path, "rb") as f:
        index_size, resident_size, count = struct.unpack("<QQQ", f.read(HDR))
        f.seek(0)
        blob = f.read(index_size)

    tensors = []
    for i in range(count):
        rec = blob[HDR + i * ENT: HDR + (i + 1) * ENT]
        name_off, name_len = struct.unpack("<IH", rec[0:6])
        dtype = rec[6]
        file_off, size = struct.unpack("<QQ", rec[8:24])
        dims = [d for d in struct.unpack("<IIII", rec[24:40]) if d > 0]
        s_off, s_size, b_off, b_size = struct.unpack("<QQQQ", rec[40:72])
        name = blob[name_off:name_off + name_len].decode()

        # dtype mapping: 0=INT4 Affine, 1=BF16, 2=FP16, 3=INT8
        dtype_str = {0: "INT4_Affine", 1: "BF16", 2: "FP16", 3: "INT8"}.get(dtype, f"dtype_{dtype}")
        tensors.append({
            "name": name,
            "dims": dims,
            "dtype": dtype_str,
            "size_bytes": size,
            "offset": file_off,
            "scale_offset": s_off if s_size > 0 else None,
            "bias_offset": b_off if b_size > 0 else None
        })
    return tensors


def main():
    repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    model_dir = sys.argv[1] if len(sys.argv) > 1 else os.path.join(repo_root, "scratch/qwen36.gturbo")
    manifest_path = os.path.join(model_dir, "manifest.json")
    weights_path = os.path.join(model_dir, "model_weights.bin")
    experts_dir = os.path.join(model_dir, "packed_experts")

    print(f"Reading model manifest from {manifest_path}...")
    with open(manifest_path, "r", encoding="utf-8") as f:
        manifest = json.load(f)

    print(f"Indexing resident tensors from {weights_path}...")
    tensors = parse_tensor_index(weights_path)

    # Classify tensors
    attention_tensors = [t for t in tensors if "attn" in t["name"] or "attention" in t["name"]]
    router_tensors = [t for t in tensors if "gate.weight" in t["name"] and "shared" not in t["name"]]
    shared_expert_tensors = [t for t in tensors if "shared_expert" in t["name"]]
    norm_tensors = [t for t in tensors if "norm" in t["name"]]

    expert_files = sorted(os.listdir(experts_dir)) if os.path.exists(experts_dir) else []

    dossier = {
        "model_id": manifest.get("modelID", "qwen36"),
        "family": manifest.get("arch", {}).get("family", "qwen36"),
        "num_layers": manifest.get("numLayers", 40),
        "experts_per_layer": manifest.get("expertsPerLayer", 256),
        "quantization_schemes": manifest.get("quant", {}),
        "total_resident_tensors": len(tensors),
        "router_tensors_count": len(router_tensors),
        "shared_expert_tensors_count": len(shared_expert_tensors),
        "expert_layers_count": len(expert_files),
        "sample_router_tensor": router_tensors[0] if router_tensors else None,
        "sample_attention_tensor": attention_tensors[0] if attention_tensors else None,
        "sample_shared_expert_tensor": shared_expert_tensors[0] if shared_expert_tensors else None,
        "all_tensors_summary": [
            {"name": t["name"], "dims": t["dims"], "dtype": t["dtype"], "size_bytes": t["size_bytes"]}
            for t in tensors
        ]
    }

    out_json = sys.argv[2] if len(sys.argv) > 2 else os.path.join(model_dir, "model_dossier.json")
    with open(out_json, "w", encoding="utf-8") as f:
        json.dump(dossier, f, indent=2)

    print(f"✅ Architectural dossier created: {out_json}")
    print(f"   Total Resident Tensors: {len(tensors)}")
    print(f"   Router Gates: {len(router_tensors)} tensors (dims: {router_tensors[0]['dims'] if router_tensors else 'N/A'})")
    print(f"   Shared Experts: {len(shared_expert_tensors)} tensors")
    print(f"   Packed Expert Layers: {len(expert_files)} files")


if __name__ == "__main__":
    main()
