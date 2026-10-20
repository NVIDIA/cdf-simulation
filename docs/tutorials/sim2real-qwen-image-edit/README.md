<!--
SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
SPDX-License-Identifier: Apache-2.0
-->

# Fine-tune Qwen-Image-Edit for SMT Sim2Real

Adapt the appearance of simulated surface-mount technology (SMT) component
crops to a chosen automated optical inspection (AOI) lighting style. Supply
your own aligned input/target images. This tutorial produces a LoRA adapter,
held-out comparisons, and optionally a full Diffusers pipeline.

Start with [the notebook](fine_tune_smt.ipynb). It explains the model, validates
data, runs a short smoke test, trains, evaluates, and performs fresh inference.
The same stages are available through `smt_lora.py` for a terminal or Slurm.

**Validation status:** the smoke workflow and a 3,000-step training run have
passed on an RTX PRO 6000 Blackwell in a fresh Python virtual environment,
including held-out inference, full-model export and offline reload. The full
run inspected three validation and three test inputs; it did not evaluate the
whole held-out dataset. A100/H100 execution and Slurm submission have not been
physically tested. This checks execution, not inspection quality. No training
images or pretrained adapters are bundled.

## Scope and model

The base is **`Qwen/Qwen-Image-Edit`**, the original single-image pipeline.
Do not substitute 2509, 2511, or `QwenImageEditPlusPipeline`; their conditioning
contracts differ. The adapter changes the diffusion transformer's attention
and feed-forward projections; the image VAE and image/text encoder stay frozen.

This tutorial trains on SMT component crops to transfer lighting and texture.
Check geometry and defect preservation in the generated images.

This trains a new style adapter from the upstream base. It does not continue
training the released CDF adapter, reproduce a published checkpoint exactly,
or train AnomalyGen or an inspection classifier. CDF's image-edit service has
its own model and serving contract; a local export is not automatically a
replacement service.

## Minimum requirements

| Requirement | Minimum or supported configuration |
|---|---|
| Operating system | Linux x86_64; Ubuntu 24.04.4 was tested. |
| Python | Python 3.12 with `venv` and `pip`. |
| GPU | **One NVIDIA GPU with at least 80 GB VRAM and native BF16 support.** Use a full GPU allocation. A100 40 GB and smaller MIG slices are outside this recipe; multiple GPUs do not combine their memory. |
| Host memory | 64 GB RAM minimum; 128 GB recommended for headroom. The tested 64 GB host also had 30 GiB swap configured, so 64 GB without swap has not been verified. |
| Free disk | At least 200 GB for dependencies, base weights, cache, checkpoints, and an optional full export; allow more for larger datasets or multiple runs. |
| NVIDIA driver | 570.124.06 or newer, with support for your specific GPU model. A newer driver may be required by that model. Driver 590.48.01 was tested. |
| Other setup | Git on `PATH`, an isolated Python environment, and internet access for initial package/model downloads. |

Use the same pinned PyTorch 2.10.0/CUDA 12.8 environment on the following
GPUs. The [PyTorch wheel build configuration](https://github.com/pytorch/pytorch/blob/v2.10.0/.ci/manywheel/build_cuda.sh#L96-L100)
includes Ampere, Hopper, and the Blackwell architectures used below. This is
software compatibility evidence; only the RTX PRO 6000 row has been tested
with this tutorial on physical hardware.

| GPU | Tutorial status |
|---|---|
| A100 **80 GB** | Compatible architecture and memory capacity; physical execution not yet tested. |
| H100 **80 GB** | Compatible architecture and memory capacity; physical execution not yet tested. |
| RTX PRO 6000 Blackwell **96 GB** | Complete smoke workflow and 3,000-step training validated. |
| Other Blackwell GPUs with **80 GB or more** | Conditional on Linux x86_64, driver/model support, a compatible pinned wheel, and passing preflight plus smoke. This is not a claim for every Blackwell system or ARM host. |

The driver baseline follows the Linux driver listed for
[CUDA 12.8 Update 1](https://docs.nvidia.com/cuda/archive/12.8.1/cuda-toolkit-release-notes/index.html).
The preflight checks native BF16, the selected device and its memory, and a
BF16 attention forward/backward operation. The model uses PyTorch's native
scaled dot-product attention (SDPA), which selects an available backend for
the GPU; no separately compiled FlashAttention package is needed.

## Set up an isolated environment

Tested on Ubuntu 24.04.4, Python 3.12.3, NVIDIA driver 590.48.01, one
**RTX PRO 6000 Blackwell with 96 GB VRAM**, and a host with 64 GB RAM plus
30 GiB configured swap. The runs used approximately 41 GiB of allocated GPU
memory during training and 58 GiB during inference. These are measured
allocations, not hardware minima; the supported recipe requires an 80 GB or
larger GPU, and other crop shapes or settings can use more memory.

Allow roughly 200 GB free disk for dependencies, base weights, cache, and an
optional full export. The downloaded base occupies about 54 GiB and a fused
export adds roughly another 54 GiB. Keep extra capacity for multiple runs.
Training, cache and comparison logs record stage timing and peak GPU memory.

Measured reference workload: 221 training pairs, seven variants per pair
(1,547 cached samples), 3,000 steps, and three validation plus three test
inputs. These numbers describe one execution check, not a dataset-size or
hardware minimum. The checkpoint is about 203 MiB.

| Stage | Measured time | Peak allocated GPU memory |
|---|---:|---:|
| Cache all 1,547 samples | 2m 45s | 16.3 GiB |
| Train 3,000 steps | 52m 38s | 40.9 GiB |
| Compare 3 held-out inputs, base and adapter | 5m 37s–5m 45s | 58.2 GiB |
| Fuse and save the full pipeline | 2m 28s | 54.3 GiB |
| Reload the fused model and predict offline | 1m 29s | 58.0 GiB |

Times are the stage runtimes recorded by the helper/validation process and
exclude initial environment installation and model download. Dataset size,
crop shapes, storage and generation settings affect runtime. Full notebook
mode evaluates all validation rows; comparison time grows with that count.
For routine software checks, use the two-pair, ten-step smoke mode.

Use a driver meeting the minimum requirements above. Confirm `nvidia-smi`
works and then run the preflight below; its attention backward operation also
checks that GPU kernels actually execute. The system CUDA toolkit and
PyTorch's bundled CUDA runtime are distinct.

Open a terminal in this tutorial folder. In `pcb-aoi`, run
`cd docs/tutorials/sim2real-qwen-image-edit`; a standalone copy of these four
files works from its own folder too. The rest of the repository is not needed
for training.

Python must include `venv` and `pip`, and Git must be on `PATH`. On a minimal
Linux installation, install your distribution's Python virtual-environment
package before these steps. The NVIDIA driver is a host prerequisite; a Python
environment does not install or replace it.

```bash
python3.12 -m venv .venv-smt
source .venv-smt/bin/activate
python -m pip install --upgrade pip
python -m pip install torch==2.10.0 torchvision==0.25.0 \
    --index-url https://download.pytorch.org/whl/cu128
python -m pip install -r requirements.txt
python -m pip check
nvidia-smi --query-gpu=index,uuid,name,memory.total --format=csv
export CUDA_VISIBLE_DEVICES=GPU-REPLACE_WITH_UUID  # Copy the chosen GPU UUID above.
python smt_lora.py preflight
python -m jupyter lab fine_tune_smt.ipynb
```

The training environment is separate from Isaac Sim and the root package's
dependencies. Git is required to install the pinned Diffusers revision.
PyTorch 2.10/CUDA 12.8 supports the listed Ampere, Hopper, and Blackwell GPUs
with one environment. Run the notebook or terminal smoke test after
installation to check your own host.

Set `CUDA_VISIBLE_DEVICES` before starting Jupyter or any GPU stage. On a
multi-GPU workstation, copy the desired GPU UUID from `nvidia-smi`; CUDA and
`nvidia-smi` can enumerate devices in different orders, so an index can select
the wrong card. The preflight prints the selected device and its available
memory. On Slurm, preserve the value provided by the scheduler rather than
overriding the allocated device.

For a remote GPU workstation, launch Jupyter there with `--no-browser
--ip=127.0.0.1 --port=8888`. On your laptop, open an SSH tunnel with
`ssh -L 8888:localhost:8888 user@gpu-host`, then open the localhost URL printed
by Jupyter. Keep Jupyter bound to localhost; the notebook kernel and all paths
belong to the GPU host.

References: [PyTorch 2.10 installation](https://pytorch.org/get-started/previous-versions/#v2100),
[Blackwell support](https://pytorch.org/blog/pytorch-2-7/),
[GPU specifications](https://www.nvidia.com/en-us/products/workstations/professional-desktop-gpus/rtx-pro-6000/).

## Bring your own data

For each pair, the input and target must show the **same crop, component,
viewpoint, and geometry**. The input is your simulation/source appearance;
the target is the desired inspection appearance. Match pixel dimensions and
spatial alignment before training. Resizing unrelated images is not registration.
Use the existing CDF rendering/cropping workflows to produce source crops;
rendered crops alone do not supply the target photographs.

Targets may be registered real photographs (`target_kind=real`) or reviewed
outputs of a teacher model you can access (`target_kind=pseudo`). Generated
targets teach the teacher's appearance and mistakes; they are not real-world
ground truth. Provide your own paired images and record how targets were created.

```text
my_smt_data/
  input/           # source RGB PNG/JPEG crops
  target/          # matching filenames, aligned target RGB crops
  pairs.csv        # one explicit record per pair
```

Create `pairs.csv` with this header and your actual filenames:

```csv
id,filename,group,split,target_kind
sample_a,component_a.png,source_board_a,train,real
sample_b,component_b.png,source_board_b,val,real
sample_c,component_c.png,source_board_c,test,real
```

These three rows illustrate the format; they are not a training dataset.
Use enough diverse crops to represent your component families, lighting, and
defects. Keep related crops from the same board/capture/render family in one
`group`, and assign the whole group to one split. A 60/20/20 group split is a
starting point, not a required ratio. The validator requires train and val;
test is recommended for a final untouched assessment. Split **before**
augmentation. Exact image duplicates across splits are rejected; near-duplicates
and semantic relationships still require your review.

The notebook converts this CSV to `pairs.jsonl`, checks matching image sizes,
rejects missing/corrupt images, repeated IDs, identity pairs, and group leakage,
and displays pairs before any GPU work. The CLI also accepts a JSONL manifest:

```json
{"id":"sample_a","input":"input/component_a.png","target":"target/component_a.png","group":"source_board_a","split":"train","target_kind":"real"}
```

Paths resolve relative to the JSONL file. Provide the val/test rows too.

## Terminal workflow

These commands work without opening the notebook. Activate the environment
above and run from the tutorial directory. Prepare `input/`, `target/`, and
`pairs.csv` as described above. Start with a fresh smoke run and set the shared
model cache explicitly. Edit the example prompt below to match your targets
before running setup.

### Prepare the run and download the base

```bash
export DATA=/absolute/path/to/my_smt_data
export RUN=/absolute/path/to/smt-smoke-001
export HF_HOME=/absolute/path/to/hf_cache
export HELPER="$PWD/smt_lora.py"
python - <<'PYSETUP'
import csv
import json
import os
from pathlib import Path
from huggingface_hub import snapshot_download
from huggingface_hub.constants import HF_HUB_OFFLINE

data = Path(os.environ["DATA"])
run = Path(os.environ["RUN"])
if not data.is_absolute() or not run.is_absolute():
    raise SystemExit("Set DATA and RUN to absolute paths.")
revision = "ac7f9318f633fc4b5778c59367c8128225f1e3de"
prompt = (
    "Render this PCB component crop as an AOI photograph under raked solder lighting. "
    "Preserve the source geometry, component shape, crop, leads, pads, markings, and defects. "
    "Use orange-red and blue specular highlights on existing exposed metal and solder. "
    "Keep matte component bodies dark and non-emissive. "
    "Do not invent text, components, halos, glow, or decorative details."
)
rows = []
with (data / "pairs.csv").open(newline="", encoding="utf-8") as stream:
    reader = csv.DictReader(stream)
    required = {"id", "filename", "group", "split", "target_kind"}
    if not required.issubset(reader.fieldnames or []):
        raise SystemExit(f"CSV needs columns: {sorted(required)}")
    for row in reader:
        filename = row["filename"].strip()
        if not filename or Path(filename).name != filename:
            raise SystemExit("Use plain filenames in flat input/target folders.")
        rows.append({
            "id": row["id"].strip(),
            "input": f"input/{filename}",
            "target": f"target/{filename}",
            "group": row["group"].strip(),
            "split": row["split"].strip(),
            "target_kind": row["target_kind"].strip(),
        })
files = {
    data / "pairs.jsonl": "".join(json.dumps(row) + "\n" for row in rows),
    run / "prompt.txt": prompt + "\n",
    run / "base-revision.txt": revision + "\n",
}
for path, contents in files.items():
    if path.exists() and path.read_text(encoding="utf-8") != contents:
        raise SystemExit(f"Existing file differs: {path}. Review it or choose a new run.")
run.mkdir(parents=True, exist_ok=True)
for path, contents in files.items():
    if not path.exists():
        path.write_text(contents, encoding="utf-8")
snapshot_download(
    "Qwen/Qwen-Image-Edit", revision=revision, local_files_only=HF_HUB_OFFLINE
)
PYSETUP
```

This setup refuses to replace a different manifest, prompt, or revision file.
Downloading the base needs internet access and acceptance of its applicable
terms. If authentication is needed, use the Hugging Face CLI locally; do not
put tokens in a notebook or job file. For a populated cache on an offline
node, set `HF_HUB_OFFLINE=1` before running setup. Keep the same `HF_HOME` for
all subsequent stages.

### Run a ten-step smoke test

Run this block first. It caches two training pairs, trains for ten steps, and
compares one held-out input against the base model. It produces a checkpoint
and two prediction images; output quality is not established by ten updates.

```bash
python "$HELPER" validate --manifest "$DATA/pairs.jsonl"
python "$HELPER" preflight
python "$HELPER" cache --manifest "$DATA/pairs.jsonl" \
    --prompt-file "$RUN/prompt.txt" --revision "$(cat "$RUN/base-revision.txt")" \
    --limit 2 --augment none --output "$RUN/cache"
python "$HELPER" train --cache "$RUN/cache" --output "$RUN/train" \
    --steps 10 --save-every 10 --rank 16 --lr 1e-4 --seed 0
python "$HELPER" compare --manifest "$DATA/pairs.jsonl" \
    --lora "$RUN/train/lora-step-00010" --output "$RUN/val-step00010" \
    --split val --limit 1
```

### Run the full recipe

After inspecting the pairs and smoke results, repeat the setup block with a
**new `RUN` path**, such as `/absolute/path/to/smt-full-001`; keep `DATA` and
`HF_HOME` unchanged. This creates a fresh prompt/revision record and reuses
the downloaded model. Then run:

```bash
python "$HELPER" validate --manifest "$DATA/pairs.jsonl"
python "$HELPER" preflight
python "$HELPER" cache --manifest "$DATA/pairs.jsonl" \
    --prompt-file "$RUN/prompt.txt" --revision "$(cat "$RUN/base-revision.txt")" \
    --augment rotshift --output "$RUN/cache"
python "$HELPER" train --cache "$RUN/cache" --output "$RUN/train" \
    --steps 3000 --save-every 500 --rank 16 --lr 1e-4 --seed 0
python "$HELPER" compare --manifest "$DATA/pairs.jsonl" \
    --lora "$RUN/train/lora-step-03000" --output "$RUN/val-step03000" --split val
```

The full comparison command processes **every validation row**. The recorded
3,000-step execution check sampled three validation and three test rows with
`--limit 3`; it did not time or qualify an entire held-out dataset. Use smoke
mode for routine command checks, and review the complete relevant validation
set before accepting an adapter.

The default training settings are batch size 1, rank/alpha 16/16, learning rate
1e-4, 3,000 optimizer steps, BF16 frozen weights, FP32 trainable adapters,
gradient checkpointing, AdamW, and saves every 500 steps. Alpha always equals
rank so the exported adapter's scaling stays unambiguous. Batch size is fixed
at 1 because crops have variable aspect ratios. No distributed training or
optimizer-resume path is provided.

Caching uses area 262,144 pixels (**0.262 MP**, approximately 512×512), with
dimensions rounded to multiples of 32. Seven paired augmentations are used:
identity, ±10° rotation, and ±5% shifts in x/y, cropped to avoid padded borders.
For example, N training pairs produce 7N cached samples. Disable augmentation
when it would invalidate markings or orientation-sensitive labels.

The native inference pipeline conditions at approximately 1 MP and defaults
to that output area, which differs from training's 0.262 MP area. Setting
output height/width alone does not change how the pipeline resizes the reference
image. Qualify inference quality at the
actual deployment resolution. Saved outputs are not silently resized back to
source pixels; label coordinates require explicit scaling and visual checks.

## Run the same steps with Slurm

Install the environment and download the pinned base revision onto storage
visible from the compute node first. The data, helper, model cache, and output
paths must all be available there. Create `smt_job.sbatch` outside the source
tree, replacing the absolute paths below:

```bash
#!/usr/bin/env bash
#SBATCH --job-name=smt-lora
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=08:00:00
#SBATCH --output=smt-%j.log
set -euo pipefail

PYTHON=/shared/cdf-simulation/docs/tutorials/sim2real-qwen-image-edit/.venv-smt/bin/python
HELPER=/shared/cdf-simulation/docs/tutorials/sim2real-qwen-image-edit/smt_lora.py
DATA=/shared/my_smt_data
RUN=/shared/results/smt-run-001
export HF_HOME=/shared/model-cache
export HF_HUB_OFFLINE=1

# RUN must already contain prompt.txt and base-revision.txt from setup.
test -s "$RUN/prompt.txt"
test -s "$RUN/base-revision.txt"
"$PYTHON" "$HELPER" validate --manifest "$DATA/pairs.jsonl"
"$PYTHON" "$HELPER" preflight
"$PYTHON" "$HELPER" cache --manifest "$DATA/pairs.jsonl" \
    --prompt-file "$RUN/prompt.txt" --revision "$(cat "$RUN/base-revision.txt")" \
    --output "$RUN/cache" --augment rotshift
"$PYTHON" "$HELPER" train --cache "$RUN/cache" --output "$RUN/train" \
    --steps 3000 --save-every 500
"$PYTHON" "$HELPER" compare --manifest "$DATA/pairs.jsonl" \
    --lora "$RUN/train/lora-step-03000" --output "$RUN/val-step03000" --split val
```

Use the **same `HF_HOME` when pre-downloading**. Account, partition, GPU type,
memory limits, and time limit are cluster-specific; the requests above are
planning values. Request one full GPU with at least 80 GB VRAM and native BF16
support. The `--mem=64G` request is host RAM; use 128 GB where available,
especially when compute nodes have no swap:

```bash
sbatch --account=YOUR_ACCOUNT --partition=YOUR_GPU_PARTITION smt_job.sbatch
squeue -u "$USER"
# Substitute the job ID returned by sbatch:
tail -f smt-JOB_ID.log
sacct -j JOB_ID --format=JobID,State,ExitCode,Elapsed,MaxRSS
# Cancel only your chosen job if necessary:
scancel JOB_ID
```

Each command waits for the preceding stage and stops on failure. Slurm is a
launch option for the same single-GPU implementation, not multi-GPU training.
The underlying validation/cache/train/compare commands were tested offline on
a workstation. Slurm submission and the scheduler commands above have not
been tested on a cluster.

## Outputs and acceptance

- `cache/cache.json` binds encoded samples to input hashes, prompt, revision,
  code, and environment. Changes require a new cache directory. A partially
  built cache is never silently reused. It also records cache-build time and
  peak GPU memory.
- `train/run.json` records settings and the preflight environment.
  `train/loss.jsonl` records observed loss, elapsed training time, and peak
  allocated GPU memory.
- `train/lora-step-*/pytorch_lora_weights.safetensors` contains adapters.
  `training.json` records the required base revision and prompt. Keep them together.
- `val-step*/` contains base and adapted predictions with identical seeds and
  `comparison.json`, including duration and peak GPU memory. Review all component
  families and defect types. Use val
  to choose the checkpoint/adapter scale, then evaluate test once.

Appearance transfer can invent board structures or remove component leads,
invalidating inspection labels. This tutorial provides a training and evaluation
workflow; it does not provide a qualified adapter or establish inspection accuracy.

Inspect geometry, lead/pad count, markings, solder highlights, body appearance,
and defect preservation. Reject outputs that repair or invent defects. A lower
flow-matching loss or a prettier target match does not prove better downstream
inspection accuracy. If targets are pseudo-targets, comparisons measure agreement
with the teacher; assess real-world claims on separate real images and an
inspection task.

After a full run and validation, optionally export its final checkpoint.
For a smoke-run export, use `lora-step-00010` instead:

```bash
python "$HELPER" fuse --lora "$RUN/train/lora-step-03000" \
    --output "$RUN/fused" --scale 1.0
```

The fused directory is a full model, much larger than the adapter. Reload with
`QwenImageEditPipeline.from_pretrained(...)` and compare a fixed input/seed
against the unfused adapter before deployment; the notebook includes an offline
reload cell. BF16 fusion need not produce bit-identical pixels. CDF serving
integration and model registration require separate implementation and validation.

## Troubleshooting

| Symptom | Action |
|---|---|
| CUDA unavailable or unsupported kernel image | Select the tutorial kernel/environment, check the driver, and use the CUDA 12.8 PyTorch wheels. Run `preflight`. |
| Out of memory | Close other GPU jobs; keep batch 1 and gradient checkpointing. Lower cache area only for a new cache/run and qualify quality again. Cache/train/inference are separate processes. Full-pipeline inference may need more memory than caching. |
| Missing or misaligned pairs | Fix names, EXIF orientation, dimensions, and registration. Do not sort two unrelated directories and pair by index. |
| Cache rejected | Choose a fresh directory after data, prompt, code, revision, or environment changes. Do not copy stale `.pt` files into it. |
| Run output already exists | Use a new run directory. Checkpoints are adapter exports, not resumable optimizer snapshots. |
| Training improves loss but damages shapes | Review targets and groups; compare earlier checkpoints/lower adapter scales on val; inspect source-target alignment. |
| Slurm cannot download weights | Pre-download the exact revision with the same shared `HF_HOME` and confirm visibility from compute nodes. |
| Inference output size differs | Native inference uses approximately 1 MP. Preserve coordinate transforms when mapping labels; inspect geometry before reusing labels. |

## Further reading

- [Qwen-Image-Edit base model](https://huggingface.co/Qwen/Qwen-Image-Edit)
- [Pinned Diffusers pipeline implementation](https://github.com/huggingface/diffusers/blob/21ba39457d0b9b72dc59ef8aec7981c947c59f6b/src/diffusers/pipelines/qwenimage/pipeline_qwenimage_edit.py)
- In a CDF Simulation checkout, see the repository root README for rendering
  setup and crop generation.
