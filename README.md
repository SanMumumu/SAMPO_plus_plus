# SAMPO++

[![Project Page](https://img.shields.io/badge/Project%20Page-SAMPO%2B%2B-lightgreen)](https://sanmumumu.github.io/SAMPO_plus_plus/)

This repository now contains a SAMPO++ continuous-latent 64x64 world-model path implemented under `SAMPO/sampo_plus/`.
The new main pipeline is:

1. external continuous VAE adapter
2. temporal planner over fine latents
3. scale-wise flow renderer
4. coarse-to-fine latent sampling and frame decoding

The legacy discrete VQ + VAR path is still present under `SAMPO/vq_model/`, `SAMPO/transformer/`, `vp/`, and `mbrl/`, but it is not the default SAMPO++ training path.

## Scope

The current rewrite covers the 64x64 robotic video prediction core path only.
The following areas are intentionally left on the legacy implementation for now:

- `vp/`
- `mbrl/`
- CoTracker motion prompt training
- old discrete checkpoint compatibility

## Installation

```bash
conda create -n SAMPO python=3.9
conda activate SAMPO
pip install -r requirements.txt
```

Optional for tests:

```bash
pip install -r requirements-dev.txt
```

To evaluate FVD, download the pretrained I3D TorchScript model into `pretrained_models/i3d/i3d_torchscript.pt`.

## External VAE Backend

Two adapter backends are supported in the new path.

- `debug`: deterministic internal fallback for code integration and smoke tests
- `lightningdit_vavae`: loads LightningDiT / VA-VAE through `tokenizer.autoencoder.AutoencoderKL`

When using LightningDiT / VA-VAE, provide:

- `--vae_repo_path`: local checkout of the LightningDiT repository
- `--vae_ckpt`: pretrained VA-VAE checkpoint path

The adapter always follows the current SAMPO++ path:

- input frames: `64x64`
- resize to `256x256` for VAE encode/decode
- latent pyramid built from the finest latent scale
- decode back to `64x64`

## Stage 1: Validate and Export the Adapter

`train_tokenizer.py` is now an adapter validation/export entrypoint.

```bash
accelerate launch train_tokenizer.py \
  --output_dir runs/adapter_debug \
  --dataset_path {path_to_npz_dataset_root} \
  --oxe_data_mixes_type select \
  --resolution 64 \
  --context_length 2 \
  --segment_length 16 \
  --train_batch_size 4 \
  --vae_backend debug \
  --latent_channels 32 \
  --latent_scales 1,2,4,8,16
```

For LightningDiT / VA-VAE:

```bash
accelerate launch train_tokenizer.py \
  --output_dir runs/adapter_vavae \
  --dataset_path {path_to_npz_dataset_root} \
  --oxe_data_mixes_type select \
  --resolution 64 \
  --context_length 2 \
  --segment_length 16 \
  --train_batch_size 4 \
  --vae_backend lightningdit_vavae \
  --vae_repo_path {path_to_LightningDiT} \
  --vae_ckpt {path_to_vavae_ckpt} \
  --latent_channels 32 \
  --latent_scales 1,2,4,8,16
```

Outputs:

- `adapter/adapter_config.json`
- `adapter/adapter_state.pt`
- `adapter_metrics.json`
- `adapter_preview.png`

## Stage 2: Train SAMPO++

`train_var.py` now trains the planner + scale-wise flow renderer.

```bash
accelerate launch train_var.py \
  --output_dir runs/sampo_pp_debug \
  --dataset_path {path_to_npz_dataset_root} \
  --oxe_data_mixes_type select \
  --resolution 64 \
  --context_length 2 \
  --segment_length 16 \
  --per_device_train_batch_size 4 \
  --per_device_eval_batch_size 4 \
  --learning_rate 1e-4 \
  --checkpointing_steps 1000 \
  --validation_steps 1000 \
  --adapter_pretrained_dir runs/adapter_debug/adapter \
  --action_conditioned \
  --action_dim 4 \
  --plan_size 512 \
  --planner_hidden_size 512 \
  --planner_num_layers 4 \
  --planner_num_heads 8 \
  --renderer_hidden_size 512 \
  --renderer_num_layers 6 \
  --renderer_num_heads 8 \
  --num_flow_steps 25 \
  --with_tracking
```

Optional evaluation flags:

- `--use_frame_metrics`
- `--use_fvd --i3d_path pretrained_models/i3d/i3d_torchscript.pt`
- `--use_rectification` (reserved for a future inference-only rectification module; current code raises if enabled)

Checkpoints are saved as:

- `checkpoint-{step}/adapter/`
- `checkpoint-{step}/world_model/`
- `checkpoint-{step}/trainer_state.pt`

## Inference

```bash
python inference/predict.py \
  --pretrained_model_name_or_path runs/sampo_pp_debug/checkpoint-1000 \
  --input_path {path_to_episode_npz} \
  --dataset_name bridge \
  --output_path outputs \
  --context_length 2 \
  --segment_length 16 \
  --resolution 64 \
  --action_conditioned \
  --num_flow_steps 25
```

The inference script now performs:

1. encode context frames into continuous latents
2. autoregressively predict one plan vector per future frame
3. sample future latents scale by scale with Euler ODE integration
4. decode the full predicted video and export side-by-side GIFs

## Programmatic API

The new main entrypoints are:

- `SAMPO.sampo_plus.ContinuousLatentAdapter`
- `SAMPO.sampo_plus.SampoPlusModel`
- `SAMPO.sampo_plus.build_sampo_plus`

The new planner interface is:

- `forward_teacher_forcing(context_fine, actions, context_length) -> plan_seq`
- `step(history_fine, past_actions, cache) -> (h_t, cache)`

## Tests

Smoke-test coverage for the new path is provided in `tests/test_sampo_plus.py`.

```bash
pytest tests/test_sampo_plus.py
```

These tests cover:

- adapter encode/decode shape contract
- latent pyramid construction
- planner teacher-forcing and step interface
- renderer SI loss backward pass
- end-to-end rollout tensor shapes

## Legacy Note

The following modules remain in the repository for the old SAMPO path and downstream integrations:

- `SAMPO/vq_model/`
- `SAMPO/transformer/`
- `SAMPO/tracker/`
- `vp/`
- `mbrl/`

They are intentionally not part of the current SAMPO++ 64x64 continuous-latent training path.

## Acknowledgement

This repository builds on ideas and components from iVideoGPT, CoTracker, FlowAR, and LightningDiT / VA-VAE.
