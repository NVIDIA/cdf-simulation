# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Run the SMT tutorial stages without importing the CDF simulation package.

Use the CLI help for the validate, preflight, cache, train, compare, and fuse
commands. Every training run starts from the original Qwen-Image-Edit model.
GPU stages run in separate processes to release memory between stages.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import math
import platform
import random
import re
import time
from pathlib import Path

from PIL import Image

BASE_MODEL = "Qwen/Qwen-Image-Edit"
BASE_REVISION = "ac7f9318f633fc4b5778c59367c8128225f1e3de"
TARGET_MODULES = [
    "to_q",
    "to_k",
    "to_v",
    "to_out.0",
    "add_q_proj",
    "add_k_proj",
    "add_v_proj",
    "to_add_out",
    "img_mlp.net.0.proj",
    "img_mlp.net.2",
    "txt_mlp.net.0.proj",
    "txt_mlp.net.2",
]
EXTENSIONS = {".png", ".jpg", ".jpeg"}


def _json_write(path: Path, data: object) -> None:
    """Write JSON atomically within the destination directory."""
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def _hash(path: Path) -> str:
    """Return the SHA-256 of a file without loading it all into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _inventory(manifest: Path) -> list[dict]:
    """Validate paired files, image sizes, split groups, and exact duplicates."""
    manifest = manifest.resolve()
    rows = []
    ids: set[str] = set()
    group_splits: dict[str, str] = {}
    image_splits: dict[str, str] = {}
    pair_hashes: set[tuple[str, str]] = set()
    for number, line in enumerate(manifest.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        required = ("id", "input", "target", "group", "split", "target_kind")
        if any(
            not isinstance(row.get(key), str) or not row[key].strip()
            for key in required
        ):
            raise ValueError(f"Row {number} needs nonempty strings: {required}")
        if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", row["id"]):
            raise ValueError(f"Unsafe sample id: {row['id']}")
        if row["id"] in ids:
            raise ValueError(f"Duplicate sample id: {row['id']}")
        ids.add(row["id"])
        if row["split"] not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split in {row['id']}")
        if row["target_kind"] not in {"real", "pseudo"}:
            raise ValueError("target_kind must be real or pseudo")
        previous = group_splits.setdefault(row["group"], row["split"])
        if previous != row["split"]:
            raise ValueError(f"Group leakage: {row['group']} occurs in two splits")
        sizes = []
        pixel_hashes = []
        for key in ("input", "target"):
            path = (manifest.parent / row[key]).resolve()
            if not path.is_file() or path.suffix.lower() not in EXTENSIONS:
                raise ValueError(f"Missing or unsupported {key} image: {path}")
            with Image.open(path) as image:
                image.load()
                if image.getexif().get(274, 1) != 1:
                    raise ValueError(
                        f"Normalize EXIF orientation before pairing: {path}"
                    )
                rgb = image.convert("RGB")
                sizes.append(rgb.size)
                if min(rgb.size) < 64:
                    raise ValueError(f"Crop smaller than 64 pixels: {path}")
                # Decoded pixels catch duplicates stored in different lossless formats.
                pixel_hash = hashlib.sha256(
                    str(rgb.size).encode() + rgb.tobytes()
                ).hexdigest()
                pixel_hashes.append(pixel_hash)
                previous = image_splits.setdefault(pixel_hash, row["split"])
                if previous != row["split"]:
                    raise ValueError(f"Identical image crosses splits: {path}")
            row[key] = str(path)
            row[key + "_sha256"] = _hash(path)
        if sizes[0] != sizes[1]:
            raise ValueError(f"Unmatched image sizes for {row['id']}: {sizes}")
        pair = (pixel_hashes[0], pixel_hashes[1])
        if pair[0] == pair[1]:
            raise ValueError(
                f"Input equals target for {row['id']}; supply a style target"
            )
        if pair in pair_hashes:
            raise ValueError(f"Duplicate training pair: {row['id']}")
        pair_hashes.add(pair)
        rows.append(row)
    if not rows:
        raise ValueError("Manifest is empty")
    if not {"train", "val"}.issubset({row["split"] for row in rows}):
        raise ValueError("Provide at least one train group and one distinct val group")
    return rows


def _dimensions(area: int, size: tuple[int, int]) -> tuple[int, int]:
    """Preserve aspect ratio approximately while snapping to multiples of 32."""
    width = math.sqrt(area * size[0] / size[1])
    height = area / width
    return max(32, round(width / 32) * 32), max(32, round(height / 32) * 32)


def _augment(image: Image.Image, tag: str) -> Image.Image:
    """Apply one paired, padding-free rotation or translation crop."""
    if tag == "id":
        return image
    width, height = image.size
    if tag.startswith("rot"):
        angle = 10 if tag == "rotp10" else -10
        sine, cosine = math.sin(math.radians(10)), math.cos(math.radians(10))
        long_side, short_side = max(width, height), min(width, height)
        if short_side <= 2 * sine * cosine * long_side:
            half = short_side / 2
            crop_w = half / sine if width >= height else half / cosine
            crop_h = half / cosine if width >= height else half / sine
        else:
            divisor = cosine**2 - sine**2
            crop_w = (width * cosine - height * sine) / divisor
            crop_h = (height * cosine - width * sine) / divisor
        rotated = image.rotate(angle, resample=Image.Resampling.BILINEAR)
        box = (
            math.ceil((width - crop_w) / 2),
            math.ceil((height - crop_h) / 2),
            math.floor((width + crop_w) / 2),
            math.floor((height + crop_h) / 2),
        )
        return rotated.crop(box).resize(image.size, Image.Resampling.BILINEAR)
    dx = round(width * {"xp": 0.05, "xn": -0.05}.get(tag, 0))
    dy = round(height * {"yp": 0.05, "yn": -0.05}.get(tag, 0))
    margin_x, margin_y = max(abs(dx), 4), max(abs(dy), 4)
    box = (margin_x + dx, margin_y + dy, width - margin_x + dx, height - margin_y + dy)
    return image.crop(box).resize(image.size, Image.Resampling.BILINEAR)


def _environment() -> dict:
    """Describe versions without recording credentials or environment variables."""
    versions = {}
    for name in (
        "torch",
        "torchvision",
        "diffusers",
        "transformers",
        "peft",
        "accelerate",
    ):
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = "not installed"
    return {"python": platform.python_version(), "packages": versions}


def _preflight() -> dict:
    """Check the selected GPU, native BF16, and attention forward/backward."""
    import torch
    from torch.nn import functional

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable. Use the Linux NVIDIA GPU environment.")
    device = torch.cuda.current_device()
    properties = torch.cuda.get_device_properties(device)
    if properties.total_memory < 80_000_000_000:
        raise RuntimeError(
            "This recipe requires one GPU with at least 80 GB VRAM. "
            f"Selected {properties.name} has {properties.total_memory / 2**30:.1f} GiB. "
            "A100 40 GB, smaller GPUs, and smaller MIG partitions are unsupported; "
            "memory is not pooled across GPUs. Check CUDA_VISIBLE_DEVICES."
        )
    if not torch.cuda.is_bf16_supported(including_emulation=False):
        raise RuntimeError("This recipe requires native BF16 support")
    capability = torch.cuda.get_device_capability(device)
    cuda_version = tuple(int(v) for v in torch.version.cuda.split(".")[:2])
    if capability[0] >= 10 and cuda_version < (12, 8):
        raise RuntimeError("Blackwell requires an appropriate CUDA 12.8+ PyTorch build")
    probe = torch.ones(
        (32, 32), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    (probe @ probe).float().mean().backward()
    query = torch.ones(
        (1, 2, 32, 128), device="cuda", dtype=torch.bfloat16, requires_grad=True
    )
    mask = torch.ones((32, 32), device="cuda", dtype=torch.bool)
    attention = functional.scaled_dot_product_attention(
        query, query, query, attn_mask=mask
    )
    attention.float().square().mean().backward()
    if not all(
        torch.isfinite(value).all().item()
        for value in (probe.grad, attention, query.grad)
    ):
        raise RuntimeError(
            "GPU BF16 matrix/attention backward produced non-finite values"
        )
    torch.cuda.synchronize()
    result = _environment() | {
        "gpu": properties.name,
        "device_index": device,
        "capability": capability,
        "cuda_runtime": torch.version.cuda,
        "compiled_architectures": torch.cuda.get_arch_list(),
        "gpu_memory_gib": round(properties.total_memory / 2**30, 1),
        "gpu_free_gib": round(torch.cuda.mem_get_info(device)[0] / 2**30, 1),
        "bf16_attention_backward": "passed",
    }
    print(json.dumps(result, indent=2))
    return result


def _new_directory(path: Path) -> None:
    """Create an unused output directory and reject reruns over old results."""
    path.mkdir(parents=True, exist_ok=False)


def _stage_runtime(started: float) -> dict:
    """Report elapsed time and peak CUDA memory for a completed stage."""
    import torch

    torch.cuda.synchronize()
    return {
        "seconds": time.monotonic() - started,
        "peak_allocated_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
        "peak_reserved_gpu_gib": torch.cuda.max_memory_reserved() / 2**30,
    }


def _base_snapshot(revision: str) -> str:
    """Resolve the pre-downloaded base locally, including on offline cluster nodes."""
    from huggingface_hub import snapshot_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    if not re.fullmatch(r"[0-9a-f]{40}", revision):
        raise ValueError("Expected an exact 40-character base-model commit SHA")
    try:
        return snapshot_download(BASE_MODEL, revision=revision, local_files_only=True)
    except LocalEntryNotFoundError as error:
        raise RuntimeError(
            "Base revision is not cached. Run the notebook's model-download cell "
            "or README download step first, using the same HF_HOME."
        ) from error


def _cache(args: argparse.Namespace) -> None:
    """Cache only training examples and bind the cache to data and model revisions."""
    rows = [row for row in _inventory(args.manifest) if row["split"] == "train"]
    if args.limit:
        rows = rows[: args.limit]
    prompt = args.prompt_file.read_text(encoding="utf-8").strip()
    if not prompt:
        raise ValueError("Prompt file is empty")
    if not re.fullmatch(r"[0-9a-f]{40}", args.revision):
        raise ValueError("Use the exact 40-character Hugging Face model commit SHA")
    signature = {
        "base_model": BASE_MODEL,
        "revision": args.revision,
        "prompt": prompt,
        "area": args.area,
        "augment": args.augment,
        "rows": rows,
        "helper_sha256": _hash(Path(__file__)),
        "environment": _environment(),
    }
    if args.output.exists():
        meta_path = args.output / "cache.json"
        if not meta_path.is_file():
            raise ValueError("Incomplete cache. Choose a fresh cache directory.")
        meta = json.loads(meta_path.read_text())
        if meta["signature"] != signature:
            raise ValueError(
                "Cache/data/settings changed. Choose a fresh cache directory."
            )
        for filename, checksum in meta["files"].items():
            file = args.output / filename
            if not file.is_file() or _hash(file) != checksum:
                raise ValueError(
                    "Incomplete or modified cache. Choose a fresh directory."
                )
        print(f"Reusing verified cache: {len(meta['files'])} samples")
        return
    _preflight()
    import torch
    from diffusers import QwenImageEditPipeline

    _new_directory(args.output)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    # The transformer is unnecessary for encoding; omit its roughly 20B weights.
    pipe = QwenImageEditPipeline.from_pretrained(
        _base_snapshot(args.revision),
        transformer=None,
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=True)
    mean = torch.tensor(pipe.vae.config.latents_mean, device="cuda").view(
        1, -1, 1, 1, 1
    )
    std = torch.tensor(pipe.vae.config.latents_std, device="cuda").view(1, -1, 1, 1, 1)
    tags = (
        ["id"]
        if args.augment == "none"
        else ["id", "rotp10", "rotn10", "xp", "xn", "yp", "yn"]
    )
    files = {}
    with torch.inference_mode():
        for index, row in enumerate(rows):
            with (
                Image.open(row["input"]) as source,
                Image.open(row["target"]) as target,
            ):
                source, target = source.convert("RGB"), target.convert("RGB")
            width, height = _dimensions(args.area, source.size)
            for tag in tags:
                images = [
                    pipe.image_processor.resize(_augment(im, tag), height, width)
                    for im in (source, target)
                ]
                latents = []
                for im in images:
                    tensor = pipe.image_processor.preprocess(im, height, width)
                    tensor = tensor.unsqueeze(2).to("cuda", torch.bfloat16)
                    latent = pipe.vae.encode(tensor).latent_dist.mode()
                    latents.append(((latent - mean.to(latent)) / std.to(latent)).cpu())
                embeds, mask = pipe.encode_prompt(
                    image=images[0], prompt=prompt, device="cuda"
                )
                if mask is None:
                    mask = torch.ones(embeds.shape[:2], dtype=torch.bool, device="cuda")
                filename = f"{row['id']}__{tag}.pt"
                torch.save(
                    {
                        "control": latents[0],
                        "target": latents[1],
                        "embeds": embeds.cpu(),
                        "mask": mask.bool().cpu(),
                        "width": width,
                        "height": height,
                    },
                    args.output / filename,
                )
                files[filename] = _hash(args.output / filename)
            print(f"Cached {index + 1}/{len(rows)} pairs", flush=True)
    _json_write(
        args.output / "cache.json",
        {"signature": signature, "files": files, "runtime": _stage_runtime(started)},
    )
    print(f"Complete: {len(files)} cached samples in {args.output}")


def _pack(latents: object) -> object:
    """Pack B,C,1,H,W latents into a sequence of two-by-two patches."""
    batch, channels, frames, height, width = latents.shape
    if frames != 1 or height % 2 or width % 2:
        raise ValueError("Expected single-frame latents with even spatial dimensions")
    return (
        latents.reshape(batch, channels, height // 2, 2, width // 2, 2)
        .permute(0, 2, 4, 1, 3, 5)
        .reshape(batch, height * width // 4, channels * 4)
    )


def _train(args: argparse.Namespace) -> None:
    """Train fresh LoRA weights with the SMT paired flow-matching objective."""
    import torch
    from diffusers import FlowMatchEulerDiscreteScheduler, QwenImageTransformer2DModel
    from peft import LoraConfig, get_peft_model_state_dict
    from safetensors.torch import save_file
    from torch.nn import functional

    environment = _preflight()
    meta = json.loads((args.cache / "cache.json").read_text())
    if meta["signature"]["helper_sha256"] != _hash(Path(__file__)):
        raise ValueError("Helper changed since caching; rebuild the cache")
    if meta["signature"]["environment"] != _environment():
        raise ValueError("Environment changed since caching; rebuild the cache")
    for filename, checksum in meta["files"].items():
        if _hash(args.cache / filename) != checksum:
            raise ValueError(f"Modified cache file: {filename}")
    files = list(meta["files"])
    if not files:
        raise ValueError("No cached training examples")
    _new_directory(args.output)
    torch.manual_seed(args.seed)
    rng = random.Random(args.seed)
    revision = meta["signature"]["revision"]
    base_path = _base_snapshot(revision)
    transformer = QwenImageTransformer2DModel.from_pretrained(
        base_path,
        subfolder="transformer",
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    transformer.requires_grad_(False)
    transformer.enable_gradient_checkpointing()
    transformer.add_adapter(
        LoraConfig(
            r=args.rank,
            lora_alpha=args.rank,
            lora_dropout=0.0,
            init_lora_weights="gaussian",
            target_modules=TARGET_MODULES,
        )
    )
    trainable = [
        parameter for parameter in transformer.parameters() if parameter.requires_grad
    ]
    for parameter in trainable:
        parameter.data = parameter.data.float()
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        base_path,
        subfolder="scheduler",
        local_files_only=True,
    )
    optimizer = torch.optim.AdamW(
        trainable, lr=args.lr, betas=(0.9, 0.99), weight_decay=1e-4
    )
    run = {
        "base_model": BASE_MODEL,
        "revision": revision,
        "rank": args.rank,
        "alpha": args.rank,
        "lr": args.lr,
        "seed": args.seed,
        "steps": args.steps,
        "batch_size": 1,
        "environment": environment,
        "cache": meta,
    }
    _json_write(args.output / "run.json", run)
    transformer.train()
    started = time.monotonic()
    step = 0
    torch.cuda.reset_peak_memory_stats()
    with (args.output / "loss.jsonl").open("w", encoding="utf-8") as log:
        while step < args.steps:
            rng.shuffle(files)
            for filename in files:
                if step >= args.steps:
                    break
                sample = torch.load(
                    args.cache / filename, map_location="cpu", weights_only=True
                )
                target = _pack(sample["target"].to("cuda", torch.bfloat16))
                control = _pack(sample["control"].to("cuda", torch.bfloat16))
                count = target.shape[1]
                config = scheduler.config
                slope = (
                    config.get("max_shift", 1.15) - config.get("base_shift", 0.5)
                ) / (
                    config.get("max_image_seq_len", 4096)
                    - config.get("base_image_seq_len", 256)
                )
                mu = config.get("base_shift", 0.5) + slope * (
                    count - config.get("base_image_seq_len", 256)
                )
                sigma = torch.rand(1, device="cuda").clamp_min(1e-6)
                sigma = math.exp(mu) / (math.exp(mu) + 1 / sigma - 1)
                noise = torch.randn_like(target)
                noisy = (1 - sigma.to(target)) * target + sigma.to(target) * noise
                height, width = sample["height"] // 16, sample["width"] // 16
                guidance = (
                    torch.ones(1, device="cuda")
                    if transformer.config.guidance_embeds
                    else None
                )
                prediction = transformer(
                    hidden_states=torch.cat([noisy, control], dim=1),
                    timestep=sigma.to(torch.bfloat16),
                    guidance=guidance,
                    encoder_hidden_states=sample["embeds"].to("cuda", torch.bfloat16),
                    encoder_hidden_states_mask=sample["mask"].to("cuda"),
                    img_shapes=[[(1, height, width), (1, height, width)]],
                    return_dict=False,
                )[0][:, :count]
                loss = functional.mse_loss(prediction.float(), (noise - target).float())
                if not torch.isfinite(loss):
                    raise RuntimeError(
                        "Non-finite loss. Inspect data and numerical settings."
                    )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable, 1.0, error_if_nonfinite=True)
                optimizer.step()
                step += 1
                record = {
                    "step": step,
                    "loss": loss.item(),
                    "seconds": time.monotonic() - started,
                    "peak_gpu_gib": torch.cuda.max_memory_allocated() / 2**30,
                }
                log.write(json.dumps(record) + "\n")
                log.flush()
                if step % 10 == 0 or step == args.steps:
                    print(json.dumps(record), flush=True)
                if step % args.save_every == 0 or step == args.steps:
                    checkpoint = args.output / f"lora-step-{step:05d}"
                    checkpoint.mkdir()
                    state = {
                        f"transformer.{key}": value.detach()
                        .to("cpu", torch.bfloat16)
                        .contiguous()
                        for key, value in get_peft_model_state_dict(transformer).items()
                    }
                    save_file(
                        state, str(checkpoint / "pytorch_lora_weights.safetensors")
                    )
                    _json_write(checkpoint / "training.json", run | {"step": step})
                    print(f"Saved {checkpoint}", flush=True)


def _load_checkpoint(path: Path) -> dict:
    """Read checkpoint provenance before loading an adapter."""
    if not (path / "pytorch_lora_weights.safetensors").is_file():
        raise ValueError("Checkpoint has no LoRA weights")
    info = json.loads((path / "training.json").read_text())
    if info["base_model"] != BASE_MODEL or info["alpha"] != info["rank"]:
        raise ValueError("Checkpoint does not match this tutorial's base/LoRA contract")
    return info


def _compare(args: argparse.Namespace) -> None:
    """Compare base and adapted outputs using identical inputs and random seeds."""
    import torch
    from diffusers import QwenImageEditPipeline

    info = _load_checkpoint(args.lora)
    all_rows = _inventory(args.manifest)
    rows = [row for row in all_rows if row["split"] == args.split]
    if not rows:
        raise ValueError(f"No {args.split} examples")
    training_rows = info["cache"]["signature"]["rows"]
    groups = {row["group"] for row in training_rows}
    hashes = {
        row[key] for row in training_rows for key in ("input_sha256", "target_sha256")
    }
    if any(
        row["group"] in groups
        or row["input_sha256"] in hashes
        or row["target_sha256"] in hashes
        for row in rows
    ):
        raise ValueError("Evaluation data overlaps this checkpoint's training data")
    if args.limit:
        rows = rows[: args.limit]
    _preflight()
    _new_directory(args.output)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    pipe = QwenImageEditPipeline.from_pretrained(
        _base_snapshot(info["revision"]),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    pipe.set_progress_bar_config(disable=False)
    prompt = info["cache"]["signature"]["prompt"]
    for label in ("base", "adapted"):
        if label == "adapted":
            pipe.load_lora_weights(
                str(args.lora),
                weight_name="pytorch_lora_weights.safetensors",
                adapter_name="smt",
            )
            pipe.set_adapters("smt", adapter_weights=args.scale)
        for row in rows:
            with Image.open(row["input"]) as image:
                image = image.convert("RGB")
            with torch.inference_mode():
                prediction = pipe(
                    image=image,
                    prompt=prompt,
                    negative_prompt=" ",
                    true_cfg_scale=args.cfg,
                    num_inference_steps=args.steps,
                    generator=torch.Generator(device="cuda").manual_seed(args.seed),
                ).images[0]
            prediction.save(args.output / f"{row['id']}__{label}.png")
    _json_write(
        args.output / "comparison.json",
        {
            "rows": rows,
            "lora": str(args.lora.resolve()),
            "scale": args.scale,
            "steps": args.steps,
            "cfg": args.cfg,
            "seed": args.seed,
            "prompt": prompt,
            "base_model": BASE_MODEL,
            "revision": info["revision"],
            "runtime": _stage_runtime(started),
            "note": "Native pipeline sizing (~1 MP); resized panels are for visual comparison only.",
        },
    )
    print(f"Compare source, target, base and adapted images in {args.output}")


def _fuse(args: argparse.Namespace) -> None:
    """Merge the adapter into a separate full pipeline with recorded provenance."""
    import torch
    from diffusers import QwenImageEditPipeline

    info = _load_checkpoint(args.lora)
    _preflight()
    _new_directory(args.output)
    started = time.monotonic()
    torch.cuda.reset_peak_memory_stats()
    pipe = QwenImageEditPipeline.from_pretrained(
        _base_snapshot(info["revision"]),
        torch_dtype=torch.bfloat16,
        local_files_only=True,
    ).to("cuda")
    pipe.load_lora_weights(
        str(args.lora),
        weight_name="pytorch_lora_weights.safetensors",
        adapter_name="smt",
    )
    pipe.fuse_lora(lora_scale=args.scale, safe_fusing=True)
    pipe.unload_lora_weights()
    pipe.save_pretrained(
        str(args.output), safe_serialization=True, max_shard_size="10GB"
    )
    _json_write(
        args.output / "training_provenance.json",
        info | {"fusion_scale": args.scale, "export_runtime": _stage_runtime(started)},
    )


def _positive(value: str) -> int:
    """Parse a positive integer for CLI counts."""
    number = int(value)
    if number <= 0:
        raise argparse.ArgumentTypeError("Expected a positive integer")
    return number


def main() -> None:
    """Run one tutorial stage from the command line."""
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    validate = sub.add_parser(
        "validate", help="Validate a paired JSONL manifest on CPU"
    )
    validate.add_argument("--manifest", type=Path, required=True)
    sub.add_parser("preflight", help="Check CUDA and execute a tiny BF16 backward pass")
    cache = sub.add_parser("cache", help="Encode training pairs on one GPU")
    cache.add_argument("--manifest", type=Path, required=True)
    cache.add_argument("--prompt-file", type=Path, required=True)
    cache.add_argument(
        "--revision", required=True, help="Exact Hugging Face base-model commit SHA"
    )
    cache.add_argument("--output", type=Path, required=True)
    cache.add_argument("--area", type=_positive, default=262144)
    cache.add_argument("--augment", choices=["none", "rotshift"], default="rotshift")
    cache.add_argument("--limit", type=_positive)
    train = sub.add_parser(
        "train", help="Train a fresh adapter; optimizer resume is not supported"
    )
    train.add_argument("--cache", type=Path, required=True)
    train.add_argument("--output", type=Path, required=True)
    train.add_argument("--steps", type=_positive, default=3000)
    train.add_argument("--save-every", type=_positive, default=500)
    train.add_argument("--rank", type=_positive, default=16)
    train.add_argument("--lr", type=float, default=1e-4)
    train.add_argument("--seed", type=int, default=0)
    compare = sub.add_parser(
        "compare", help="Generate held-out base/adapter comparison images"
    )
    compare.add_argument("--manifest", type=Path, required=True)
    compare.add_argument("--lora", type=Path, required=True)
    compare.add_argument("--output", type=Path, required=True)
    compare.add_argument("--split", choices=["val", "test"], default="val")
    compare.add_argument("--limit", type=_positive)
    compare.add_argument("--steps", type=_positive, default=30)
    compare.add_argument("--cfg", type=float, default=4.0)
    compare.add_argument("--seed", type=int, default=0)
    compare.add_argument("--scale", type=float, default=1.0)
    fuse = sub.add_parser("fuse", help="Export a separate full Diffusers pipeline")
    fuse.add_argument("--lora", type=Path, required=True)
    fuse.add_argument("--output", type=Path, required=True)
    fuse.add_argument("--scale", type=float, default=1.0)
    args = parser.parse_args()
    if args.command == "validate":
        rows = _inventory(args.manifest)
        print(
            json.dumps(
                {
                    split: sum(row["split"] == split for row in rows)
                    for split in ("train", "val", "test")
                },
                indent=2,
            )
        )
    elif args.command == "preflight":
        _preflight()
    else:
        if args.command == "train" and (not math.isfinite(args.lr) or args.lr <= 0):
            parser.error("--lr must be finite and positive")
        {"cache": _cache, "train": _train, "compare": _compare, "fuse": _fuse}[
            args.command
        ](args)


if __name__ == "__main__":
    main()
