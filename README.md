# GPT-2 Continual Learning Patch (Reversible Sidecars)

This repository implements a practical continual-learning recipe for GPT-2 that:

- **Keeps the pretrained base weights fixed** (preserves the representation “coordinate system”).
- **Adds a small trainable patch** after each transformer block.
- Makes the patch **invertible** (coupling transform) and the updates **reversible** (patch-only ledgers + rollback).
- Uses **replay + distillation** to constrain drift on old behavior.

The goal is not “perfect lifelong learning”; it is a **surgical, controllable** way to add new capability while being able to undo it.

## Philosophy (core principles)

1. **Preserve a stable coordinate system**  
   Catastrophic forgetting is largely caused by moving the latent representation map. Operationally:
   - Base parameters **θ**: frozen (or nearly frozen)
   - Patch parameters **φ**: small, trainable, per-layer

2. **Add capability as a removable perturbation**  
   Instead of changing θ, attach a per-layer sidecar that can be turned off or removed without touching θ.

3. **Localize change (routing + smallness)**  
   A gate per layer controls activation so the patch does not “fire everywhere”. Regularize gates/patch magnitude.

4. **Constrain drift in function space**  
   Use replay and logit distillation (KL) against a teacher (last committed patch) rather than weight-space heuristics.

5. **Make learning reversible at the level that matters**  
   - *Patch invertibility*: the coupling map is bijective.
   - *Training reversibility*: store patch-only checkpoints and rollback on regressions.

## Mathematical model

### Base GPT-2

Let `x` be a token sequence, and a pretrained causal LM have parameters θ:

```
h0 = E(x)
h(l+1) = B_l(h_l; θ)     for l = 0..L-1
logits = W h_L
```

### Patched model

We keep θ frozen and insert a patch after each block:

```
u_l = B_l(h_l; θ)
h(l+1) = u_l + g_l(u_l) · S_l(u_l; φ_l)
```

Where:
- `g_l(u_l) ∈ (0,1)` is a gate (routing / activation control)
- `S_l(·)` is a bounded “sidecar” delta map (the learned perturbation)

In code, this is implemented in `cl_patch_gpt2.PatchedGPT2Block`.

## The invertible coupling patch

We use an **additive coupling transform** on the last dimension. For a hidden vector `h ∈ R^d` with even `d`,
split channels into halves:

```
h = (a, b)      a, b ∈ R^{d/2}
```

Define two functions `F, G : R^{d/2} → R^{d/2}` (small MLPs) and a scalar scale `s > 0`:

Forward coupling:
```
a' = a + s · F(b)
b' = b + s · G(a')
```

Inverse (closed form, always exists):
```
b  = b' − s · G(a')
a  = a' − s · F(b)
```

So the coupling map `C(h) = (a', b')` is bijective for **any** `F, G` and `s`.

To turn this into a sidecar residual, define:
```
S(h) = C(h) − h
```

And apply it with a gate:
```
h_next = u + g(u) · (C(u) − u)
```

### Near-identity initialization

To start close to the pretrained model:
- Initialize the last projection of `F` and `G` to zeros (so `F ≈ 0`, `G ≈ 0` at init).
- Use a small initial scale `s` (see `init_scale`).

Then `C(h) ≈ h`, `S(h) ≈ 0`.

### Gate choice

We use a simple sequence-level gate:
```
g(u) = sigmoid(wᵀ mean_t u_t + b)
```

Regularize gate sparsity/smallness (e.g., L1 on `g`) to keep most patches mostly inactive.

Implemented as `cl_patch_gpt2.SeqGate`.

## Continual learning objective

At phase `t`, train only `φ` on new data `D_t` while constraining behavior on replay `R_t`.
The typical objective is:

```
L(φ) =
  L_LM(x ~ D_t)
  + α · L_LM(x' ~ R_t)
  + β · KL( teacher(x') || student(x') )
  + γ · R_patch
  + η · R_gate
```

This repo implements:
- `L_LM` on the new stream and replay (`replay_loss_weight`)
- Logit-level distillation KL against the last committed teacher (`distill_kl_weight`, `distill_temperature`)
- Regularizers:
  - `gate_l1_weight`: mean gate magnitude
  - `patch_l2_weight`: mean squared patch delta magnitude
  - `scale_l2_weight`: L2 penalty on `log_scale` (keeps `s` bounded)

Optional hidden-statistics stabilization (μ/Σ drift) is not implemented here, but can be added by hooking block outputs.

## Reversibility (what it means here)

There are two distinct reversibility claims:

1. **Invertible patch map**  
   The coupling transform `C` is bijective and has an explicit inverse. The training script performs a small
   sanity check at startup (`check_invertibility=true`): `inverse(C(x)) ≈ x`.

2. **Reversible updates (ledger + rollback)**  
   Training logs patch-only checkpoints:
   - `*_start.pt`: patch at phase start
   - `*_commit.pt`: patch committed at phase end
   - `*_good_step*.pt`: last “good” patch after probe eval
   - `*_rollback_step*.pt`: rollback snapshots when probes regress

Rollback triggers when probe loss increases beyond a tolerance, then restores the last good patch state.
This is implemented in `train_continual_gpt2.py`.

## What to measure (validation checklist)

If the approach is behaving as intended, you should see:

1. **Plasticity**: new-phase LM loss decreases.
2. **Stability**: probe loss on old data stays within tolerance (or rollbacks recover it).
3. **Locality**: gates are mostly small (depends on your gate regularization).
4. **Patch magnitude**: patch L2 and scale `s` remain bounded.
5. **Ledger usefulness**: rollbacks restore probe metrics and training continues.

## Repo layout

- `cl_patch_gpt2.py`: reversible coupling sidecar, gate, GPT-2 block wrapper, patch-only save/load.
- `cl_data.py`: streaming text iterators, token block dataset, replay buffer.
- `train_continual_gpt2.py`: continual learning loop with replay + distill + rollback + logging.
- Configs:
  - `config_gpt2_example.json`: HF streaming example (wikitext → imdb).
  - `config_gpt2_smoke_files.json`: tiny 2-phase local smoke run.
  - `config_gpt2_1k_files.json`: 1k-step local verification run (with probes/rollbacks).
  - `config_gpt2_100k_hf.json`: 100k-step long run (2×50k phases, repeats streams).
- Local data:
  - `data/phase1.txt`, `data/phase2.txt`: tiny two-phase streams for runtime learning demos.

## Outputs (where runs go)

All training outputs are written under `output_dir` (from your config). By default, configs in this repo point to `runs_user/...`,
which is gitignored. See `docs/RUNS.md:1` for the full ledger/log layout and how to interpret run metrics.

### Latest validated long run (example)

Using `config_gpt2_100k_hf.json`, a full `100000`-step run (50k wikitext → 50k imdb) produced patch-only checkpoints and a complete training log:

- `runs_user/gpt2_100k_hf/ledger/000_wikitext_commit.pt`
- `runs_user/gpt2_100k_hf/ledger/001_imdb_commit.pt`
- `runs_user/gpt2_100k_hf/ledger/001_imdb_good_step000100000.pt` (last “known good” patch at end of run)
- Probe checkpoints/rollbacks under `runs_user/gpt2_100k_hf/ledger/`
- `runs_user/gpt2_100k_hf/train.log` (prints `Done.` at completion)

These files are intentionally **not tracked by git**.

## Quick start (no Docker)

```bash
python -m pip install --upgrade pip
python -m pip install transformers datasets accelerate evaluate
python train_continual_gpt2.py --config config_gpt2_smoke_files.json
```

## Decoding (avoid repetition)

Greedy decoding (`do_sample=false`) can get stuck in loops (especially after continual fine-tuning).
For qualitative inspection, prefer nucleus sampling + repetition controls. The sampler script defaults to sampling:

```bash
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.home -e HF_HOME=/workspace/.hf_home \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  bash -lc 'python scripts/sample_100k_outputs.py --patch-ckpt runs_user/gpt2_100k_hf/ledger/001_imdb_commit.pt'
```

If you want deterministic greedy output (more repetition risk), add `--greedy`.

To try your own prompts:

```bash
python scripts/sample_100k_outputs.py --prompt "Review: This film was terrible. The acting was"
```

## Run the patched model

Use `scripts/run_patched_gpt2.py` to load a patch checkpoint and generate text.

```bash
docker run --rm --gpus all --ipc=host \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.home -e HF_HOME=/workspace/.hf_home \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  bash -lc 'python scripts/run_patched_gpt2.py --patch-ckpt runs_user/gpt2_100k_hf/ledger/001_imdb_commit.pt --prompt "Review: This film was terrible. The acting was"'
```

Useful options:

- Disable the patch (baseline behavior): `--gate off`
- Force patch on everywhere: `--gate on`
- Deterministic but more repetition risk: `--greedy`

## Running in NVIDIA NGC Docker (recommended)

This project was validated inside `nvcr.io/nvidia/pytorch:25.09-py3` (ARM64 build available).

### One-shot interactive run

```bash
docker run --gpus all -it --rm \
  --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.home \
  -e HF_HOME=/workspace/.hf_home \
  -e PIP_CACHE_DIR=/workspace/.pip_cache \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  bash
```

Inside the container:

```bash
python -m pip install --user transformers datasets accelerate evaluate
python train_continual_gpt2.py --config config_gpt2_1k_files.json
```

### Long run that survives SSH disconnect

Start training in a detached container and log to a file on the host:

```bash
docker run -d --name gpt2_cl_100k --restart=no \
  --gpus all --ipc=host --ulimit memlock=-1 --ulimit stack=67108864 \
  --user "$(id -u):$(id -g)" \
  -e HOME=/workspace/.home \
  -e HF_HOME=/workspace/.hf_home \
  -e PIP_CACHE_DIR=/workspace/.pip_cache \
  -e TOKENIZERS_PARALLELISM=false \
  -e PYTHONUNBUFFERED=1 \
  -v "$PWD":/workspace -w /workspace \
  nvcr.io/nvidia/pytorch:25.09-py3 \
  bash -lc 'python train_continual_gpt2.py --config config_gpt2_100k_hf.json 2>&1 | tee -a runs_user/gpt2_100k_hf/train.log'
```

Monitor:

```bash
docker logs -f gpt2_cl_100k
tail -f runs_user/gpt2_100k_hf/train.log
```

Stop:

```bash
docker stop gpt2_cl_100k
```

If you previously used `--restart unless-stopped`, disable it to avoid unintended reruns:

```bash
docker update --restart=no gpt2_cl_100k
```

## Troubleshooting notes

- **Iterable datasets end early**: set `"repeat": true` in the phase config (already done in the long-run configs).
- **Gradient checkpointing warning (“inputs have requires_grad=False”)**: this can happen with patch-only training under
  re-entrant checkpointing. This repo enables non-reentrant checkpointing when available; otherwise disable checkpointing.
- **Cache permissions**: when using Docker, run with `--user "$(id -u):$(id -g)"` and set `HF_HOME` to a mounted path.

## Scope / limitations

This is a research-oriented scaffold:
- Gates may not be sparse without stronger regularization or a router.
- Stability is only guaranteed on the chosen probe/replay distribution (not “all previous knowledge”).
- Extending to other architectures (LLaMA/Qwen) is mostly wiring (layer list + hidden size + forward signature), but needs revalidation.
