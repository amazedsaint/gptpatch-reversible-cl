# Runs and Artifacts

This project writes **patch-only** checkpoints (not full GPT-2 weights) and training logs into an `output_dir` you set in a config JSON.

Recommended convention:

- Use `output_dir` under `runs_user/` (gitignored).
- Keep all “real” training output out of the repo history.

## Directory layout

When you run `train_continual_gpt2.py`, it creates:

```
<output_dir>/
  ledger/
    000_<phase>_start.pt
    000_<phase>_commit.pt
    000_<phase>_good_step000001000.pt
    000_<phase>_rollback_step000001234.pt
    ...
```

These `.pt` files contain only:

- `patch`: a state dict of trainable patch parameters (`gate` + `sidecar` weights/scales)
- some metadata (`global_step`, `phase_idx`, metrics) depending on the checkpoint type

Because base GPT-2 weights remain frozen, patch-only checkpoints stay small (typically single-digit MB).

## What the ledger files mean

- `*_start.pt`: patch snapshot at the beginning of a phase.
- `*_good_step*.pt`: “last known good” patch after a probe evaluation passes.
- `*_rollback_step*.pt`: snapshot written when probes regress and the patch is rolled back.
- `*_commit.pt`: patch snapshot at the end of a phase (used to create the next phase’s teacher).

## Logs and progress

If you run in detached Docker, pipe stdout/stderr to a host log file (example below).
To see progress:

- `tail -f <output_dir>/train.log`
- or `docker logs -f <container_name>`

The training script prints lines like:

```
[train] step=12300 loss=... new=... old=... kl=... gate=... patch=... scale=...
```

Interpretation:

- `new`: LM loss on the current stream minibatch (plasticity).
- `old`: LM loss on a replay minibatch (stability pressure).
- `kl`: distillation KL on replay vs the teacher (function drift control).
- `gate`: mean gate activation (locality; smaller is more “patch off”).
- `patch`: mean squared magnitude of the gated patch delta (boundedness).
- `scale`: average coupling scale (keeps coupling close to identity when small).

## Preventing “restart reruns” in Docker

If you run Docker with `--restart unless-stopped` and the training script exits successfully,
Docker will restart the container and it will start training again from the beginning.

Options:

- Don’t set a restart policy for one-shot training (`--restart=no`).
- Or disable restart after it starts: `docker update --restart=no <container>`.
- Or remove the container after completion: `docker rm -f <container>`.

If a restart/rerun happens anyway, you may see multiple “run headers” in the same `train.log` (e.g., repeated
`[invertibility]` / `=== Phase 0` blocks). Treat each header as a separate run invocation.

## Example: completed 100k-step run

Using `config_gpt2_100k_hf.json`, a full run writes into:

- `runs_user/gpt2_100k_hf/train.log`
- `runs_user/gpt2_100k_hf/ledger/`

Key files to look for:

- `runs_user/gpt2_100k_hf/ledger/000_wikitext_commit.pt` (phase 0 committed patch)
- `runs_user/gpt2_100k_hf/ledger/001_imdb_commit.pt` (phase 1 committed patch)
- `runs_user/gpt2_100k_hf/ledger/001_imdb_good_step000100000.pt` (final “good” patch snapshot)
- `runs_user/gpt2_100k_hf/ledger/001_imdb_rollback_step000054000.pt` (example rollback snapshot)
