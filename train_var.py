import argparse
import json
import math
import os
import time
from pathlib import Path

import torch
from accelerate import Accelerator
from accelerate.logging import get_logger
from accelerate.utils import DistributedDataParallelKwargs, ProjectConfiguration, set_seed
from PIL import Image
from torch.optim import AdamW
from tqdm.auto import tqdm

from sampo_pp.data import DATASET_NAMED_MIXES, SimpleRoboticDataLoaderv2
from sampo_pp import (
    AdapterConfig,
    ContinuousLatentAdapter,
    FlowRendererConfig,
    SampoPlusConfig,
    SampoPlusModel,
    TemporalPlannerConfig,
    parse_scales,
)
from sampo_pp.utils.video_metric import Evaluator, FeatureStats


logger = get_logger(__name__)


def parse_float_list(value: str, length: int):
    if value is None:
        return tuple(1.0 for _ in range(length))
    weights = tuple(float(item.strip()) for item in value.split(',') if item.strip())
    if len(weights) != length:
        raise ValueError(f'Expected {length} flow loss weights, got {len(weights)}')
    return weights


def parse_args():
    parser = argparse.ArgumentParser(description='Train SAMPO++ with continuous latents and scale-wise flow matching.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--mixed_precision', type=str, default='no', choices=['no', 'fp16', 'bf16'])
    parser.add_argument('--report_to', type=str, default='tensorboard')
    parser.add_argument('--with_tracking', action='store_true')
    parser.add_argument('--resume_from_checkpoint', type=str, default=None)
    parser.add_argument('--checkpointing_steps', type=int, default=1000)
    parser.add_argument('--validation_steps', type=int, default=1000)
    parser.add_argument('--log_steps', type=int, default=50)
    parser.add_argument('--max_train_steps', type=int, default=10000)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--per_device_eval_batch_size', type=int, default=4)
    parser.add_argument('--learning_rate', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=0.01)
    parser.add_argument('--max_grad_norm', type=float, default=1.0)

    parser.add_argument('--dataset_path', type=str, default='/data2/frame_datasets')
    parser.add_argument('--dataset_size', type=int, default=None)
    parser.add_argument('--sthsth_root_path', type=str, default='/data/something-something-v2/20bn-something-something-v2-frames-64')
    parser.add_argument('--oxe_data_mixes_type', type=str, default='select')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--context_length', type=int, default=2)
    parser.add_argument('--segment_length', type=int, default=16)
    parser.add_argument('--video_stepsize', type=int, default=1)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--strong_aug', action='store_true')
    parser.add_argument('--no_aug', action='store_true')
    parser.add_argument('--action_conditioned', action='store_true')
    parser.add_argument('--action_dim', type=int, default=4)

    parser.add_argument('--adapter_pretrained_dir', type=str, default=None)
    parser.add_argument('--vae_backend', type=str, default='debug', choices=['debug', 'lightningdit_vavae'])
    parser.add_argument('--vae_ckpt', type=str, default=None)
    parser.add_argument('--vae_repo_path', type=str, default=None)
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--latent_scales', type=str, default='1,2,4,8,16')
    parser.add_argument('--model_resolution', type=int, default=256)
    parser.add_argument('--use_variational', action='store_true')

    parser.add_argument('--plan_size', type=int, default=512)
    parser.add_argument('--planner_hidden_size', type=int, default=512)
    parser.add_argument('--planner_num_layers', type=int, default=4)
    parser.add_argument('--planner_num_heads', type=int, default=8)
    parser.add_argument('--planner_mlp_ratio', type=float, default=4.0)
    parser.add_argument('--renderer_hidden_size', type=int, default=512)
    parser.add_argument('--renderer_num_layers', type=int, default=6)
    parser.add_argument('--renderer_num_heads', type=int, default=8)
    parser.add_argument('--renderer_mlp_ratio', type=float, default=4.0)
    parser.add_argument('--num_flow_steps', type=int, default=25)
    parser.add_argument('--flow_loss_weights', type=str, default=None)
    parser.add_argument('--use_rectification', action='store_true')
    # --- ablation / objective flags (one flag per ablation) ---
    parser.add_argument('--multi_scale', default=True, action=argparse.BooleanOptionalAction,
                        help='multi-scale planner (--no-multi_scale for the single-state baseline)')
    parser.add_argument('--action_mode', type=str, default='acvf',
                        choices=['concat', 'crossattn', 'adaln', 'adm', 'acvf'])
    parser.add_argument('--rope_mode', type=str, default='pcd',
                        choices=['learned', 'rope2d', 'spacetime', 'sarope4d', 'pcd'])
    parser.add_argument('--lambda_noop', type=float, default=0.1)
    parser.add_argument('--lambda_cf', type=float, default=0.1)
    parser.add_argument('--cf_margin', type=float, default=0.1)
    parser.add_argument('--rollout_aware_p', type=float, default=0.0,
                        help='probability of scheduled self-forcing (Phase 5); 0 = teacher forcing')
    parser.add_argument('--rollout_k', type=int, default=4)
    parser.add_argument('--rollout_aware_warmup', type=int, default=0)
    parser.add_argument('--lambda_rollout', type=float, default=1.0)

    parser.add_argument('--max_eval_iters', type=int, default=50)
    parser.add_argument('--use_frame_metrics', action='store_true')
    parser.add_argument('--use_fvd', action='store_true')
    parser.add_argument('--i3d_path', type=str, default=None)
    return parser.parse_args()


def build_dataloader(args, train: bool):
    if args.strong_aug:
        augmentation_args = {
            'brightness': [0.6, 1.4],
            'contrast': [0.6, 1.4],
            'saturation': [0.6, 1.4],
            'hue': [-0.5, 0.5],
            'random_resized_crop_scale': (0.6, 1.0),
            'random_resized_crop_ratio': (0.75, 1.3333),
            'no_aug': args.no_aug,
        }
    else:
        augmentation_args = {
            'brightness': [0.9, 1.1],
            'contrast': [0.9, 1.1],
            'saturation': [0.9, 1.1],
            'hue': [-0.05, 0.05],
            'random_resized_crop_scale': (0.8, 1.0),
            'random_resized_crop_ratio': (0.9, 1.1),
            'no_aug': args.no_aug,
        }
    segment_args = {
        'random_selection': False,
        'random_shuffle': False,
        'goal_conditioned': False,
        'segment_length': args.segment_length,
        'context_length': args.context_length,
        'stepsize': args.video_stepsize,
        'segment_horizon': None,
    }
    batch_size = args.per_device_train_batch_size if train else args.per_device_eval_batch_size
    return SimpleRoboticDataLoaderv2(
        parent_dir=args.dataset_path,
        datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
        batch_size=batch_size,
        num_workers=args.dataloader_num_workers,
        train=train,
        maxsize=args.dataset_size,
        image_size=args.resolution,
        sthsth_root_path=args.sthsth_root_path,
        load_action=args.action_conditioned,
        **augmentation_args,
        **segment_args,
    )


def build_adapter(args, device):
    if args.adapter_pretrained_dir:
        adapter = ContinuousLatentAdapter.from_pretrained(args.adapter_pretrained_dir)
    else:
        adapter = ContinuousLatentAdapter(
            AdapterConfig(
                backend=args.vae_backend,
                ckpt_path=args.vae_ckpt,
                repo_path=args.vae_repo_path,
                input_resolution=args.resolution,
                model_resolution=args.model_resolution,
                latent_resolution=max(parse_scales(args.latent_scales)),
                latent_channels=args.latent_channels,
                latent_scales=parse_scales(args.latent_scales),
                use_variational=args.use_variational,
            )
        )
    return adapter.to(device).freeze()


def build_model(args, adapter):
    scales = tuple(int(scale) for scale in adapter.latent_scales)
    config = SampoPlusConfig(
        context_length=args.context_length,
        latent_scales=scales,
        flow_loss_weights=parse_float_list(args.flow_loss_weights, len(scales)),
        planner=TemporalPlannerConfig(
            latent_channels=adapter.latent_channels,
            hidden_size=args.planner_hidden_size,
            plan_size=args.plan_size,
            action_dim=args.action_dim if args.action_conditioned else 0,
            num_layers=args.planner_num_layers,
            num_heads=args.planner_num_heads,
            mlp_ratio=args.planner_mlp_ratio,
            max_frames=max(args.segment_length + 4, 32),
            latent_scales=scales,
            multi_scale=args.multi_scale,
        ).__dict__,
        renderer=FlowRendererConfig(
            latent_channels=adapter.latent_channels,
            plan_size=args.plan_size,
            action_dim=args.action_dim if args.action_conditioned else 0,
            hidden_size=args.renderer_hidden_size,
            num_layers=args.renderer_num_layers,
            num_heads=args.renderer_num_heads,
            mlp_ratio=args.renderer_mlp_ratio,
            max_frames=max(args.segment_length + 4, 32),
            max_scales=max(len(scales) + 2, 8),
            action_mode=args.action_mode,
            rope_mode=args.rope_mode,
        ).__dict__,
        lambda_noop=args.lambda_noop,
        lambda_cf=args.lambda_cf,
        cf_margin=args.cf_margin,
        action_conditioned=args.action_conditioned,
        rollout_aware_p=args.rollout_aware_p,
        rollout_k=args.rollout_k,
        rollout_aware_warmup=args.rollout_aware_warmup,
        lambda_rollout=args.lambda_rollout,
    )
    return SampoPlusModel(config)


def split_batch(batch, action_conditioned, device):
    if action_conditioned:
        pixel_values, actions = batch
        return pixel_values.to(device, non_blocking=True), actions.to(device, non_blocking=True)
    return batch.to(device, non_blocking=True), None


def save_preview(gt: torch.Tensor, pred: torch.Tensor, save_path: Path):
    gt = gt[0].detach().cpu().clamp(0.0, 1.0)
    pred = pred[0].detach().cpu().clamp(0.0, 1.0)
    rows = []
    for source, target in zip(gt, pred):
        left = Image.fromarray((source.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        right = Image.fromarray((target.permute(1, 2, 0).numpy() * 255).astype('uint8'))
        canvas = Image.new('RGB', (left.size[0] * 2, left.size[1]))
        canvas.paste(left, (0, 0))
        canvas.paste(right, (left.size[0], 0))
        rows.append(canvas)
    merged = Image.new('RGB', (rows[0].size[0], rows[0].size[1] * len(rows)))
    for idx, row in enumerate(rows):
        merged.paste(row, (0, idx * row.size[1]))
    merged.save(save_path)


def save_checkpoint(output_dir, accelerator, model, adapter, optimizer, step):
    checkpoint_dir = Path(output_dir) / f'checkpoint-{step}'
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    accelerator.unwrap_model(model).save_pretrained(checkpoint_dir / 'world_model')
    adapter.save_pretrained(checkpoint_dir / 'adapter')
    state = {
        'optimizer': optimizer.state_dict(),
        'step': step,
    }
    torch.save(state, checkpoint_dir / 'trainer_state.pt')
    latest_path = Path(output_dir) / 'latest_checkpoint.txt'
    latest_path.write_text(str(checkpoint_dir), encoding='utf-8')
    return checkpoint_dir


def resolve_resume_path(output_dir, resume_from_checkpoint):
    if not resume_from_checkpoint:
        return None
    if resume_from_checkpoint != 'latest':
        return Path(resume_from_checkpoint)
    latest = Path(output_dir) / 'latest_checkpoint.txt'
    if latest.exists():
        return Path(latest.read_text(encoding='utf-8').strip())
    checkpoints = sorted(Path(output_dir).glob('checkpoint-*'), key=lambda p: int(p.name.split('-')[-1]))
    return checkpoints[-1] if checkpoints else None


@torch.no_grad()
def evaluate(args, accelerator, adapter, model, eval_dataloader, evaluator, completed_steps):
    model.eval()
    mse_values = []
    psnr_values, ssim_values, lpips_values = [], [], []
    real_feats, gen_feats = FeatureStats(capture_mean_cov=True), FeatureStats(capture_mean_cov=True)
    progress = tqdm(range(args.max_eval_iters), disable=not accelerator.is_local_main_process, desc='validation')
    for batch_index, batch in enumerate(eval_dataloader):
        if batch_index >= args.max_eval_iters:
            break
        pixel_values, actions = split_batch(batch, args.action_conditioned, accelerator.device)
        rollout = accelerator.unwrap_model(model).rollout(
            adapter,
            context_frames=pixel_values[:, : args.context_length],
            actions=actions,
            num_future_frames=args.segment_length - args.context_length,
            num_flow_steps=args.num_flow_steps,
            use_rectification=args.use_rectification,
        )
        pred = rollout['frames']
        mse = torch.mean((pred - pixel_values) ** 2)
        mse_values.append(accelerator.gather(mse.unsqueeze(0)).mean().item())

        if evaluator is not None and args.use_frame_metrics:
            metrics = evaluator(pixel_values.clamp(0.0, 1.0), pred.clamp(0.0, 1.0))
            psnr_values.append(accelerator.gather(metrics[1].unsqueeze(0)).mean().item())
            ssim_values.append(accelerator.gather(metrics[2].unsqueeze(0)).mean().item())
            lpips_values.append(accelerator.gather(metrics[3].unsqueeze(0)).mean().item())

        if evaluator is not None and args.use_fvd and evaluator.i3d_model is not None:
            detector_kwargs = dict(rescale=True, resize=True, return_features=True)
            real_feat = evaluator.i3d_model(pixel_values.permute(0, 2, 1, 3, 4).contiguous() * 255.0, **detector_kwargs)
            gen_feat = evaluator.i3d_model(pred.permute(0, 2, 1, 3, 4).contiguous() * 255.0, **detector_kwargs)
            real_feats.append_torch(accelerator.gather(real_feat))
            gen_feats.append_torch(accelerator.gather(gen_feat))

        if batch_index == 0 and accelerator.is_main_process:
            preview_dir = Path(args.output_dir) / 'images'
            preview_dir.mkdir(parents=True, exist_ok=True)
            save_preview(pixel_values, pred, preview_dir / f'val-preview-{completed_steps}.png')

        progress.update(1)

    logs = {'eval/mse': sum(mse_values) / max(len(mse_values), 1)}
    if psnr_values:
        logs['eval/psnr'] = sum(psnr_values) / len(psnr_values)
        logs['eval/ssim'] = sum(ssim_values) / len(ssim_values)
        logs['eval/lpips'] = sum(lpips_values) / len(lpips_values)
    if args.use_fvd and evaluator is not None and evaluator.i3d_model is not None and real_feats.num_items > 0:
        logs['eval/fvd'] = evaluator.compute_fvd(real_feats, gen_feats)
    if args.with_tracking:
        accelerator.log(logs, step=completed_steps)
    model.train()
    return logs if accelerator.is_main_process else None


def main():
    args = parse_args()
    os.makedirs(args.output_dir, exist_ok=True)

    ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
    project_config = ProjectConfiguration(project_dir=args.output_dir, logging_dir=os.path.join(args.output_dir, 'logs'))
    accelerator = Accelerator(
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        mixed_precision=args.mixed_precision,
        log_with=args.report_to if args.with_tracking else None,
        project_config=project_config,
        kwargs_handlers=[ddp_kwargs],
    )
    logger.info(accelerator.state, main_process_only=False)
    if args.seed is not None:
        set_seed(args.seed, device_specific=True)

    adapter = build_adapter(args, accelerator.device)
    model = build_model(args, adapter)
    optimizer = AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    train_dataloader = build_dataloader(args, train=True)
    eval_dataloader = build_dataloader(args, train=False)
    if args.use_frame_metrics or args.use_fvd:
        evaluator = Evaluator(i3d_path=args.i3d_path if args.use_fvd else None).to(accelerator.device)
    else:
        evaluator = None

    model, optimizer, train_dataloader, eval_dataloader = accelerator.prepare(
        model,
        optimizer,
        train_dataloader,
        eval_dataloader,
    )

    completed_steps = 0
    resume_path = resolve_resume_path(args.output_dir, args.resume_from_checkpoint)
    if resume_path is not None and (resume_path / 'world_model').exists():
        state = torch.load(resume_path / 'trainer_state.pt', map_location='cpu')
        accelerator.unwrap_model(model).load_state_dict(torch.load(resume_path / 'world_model' / 'model.pt', map_location='cpu'))
        optimizer.load_state_dict(state['optimizer'])
        completed_steps = int(state['step'])
        logger.info(f'Resumed from {resume_path}')

    if args.with_tracking:
        accelerator.init_trackers('sampo_pp', vars(args))

    progress_bar = tqdm(range(completed_steps, args.max_train_steps), disable=not accelerator.is_local_main_process, desc='steps')
    model.train()
    train_iterator = iter(train_dataloader)
    while completed_steps < args.max_train_steps:
        try:
            batch = next(train_iterator)
        except StopIteration:
            train_iterator = iter(train_dataloader)
            batch = next(train_iterator)

        pixel_values, actions = split_batch(batch, args.action_conditioned, accelerator.device)
        with accelerator.accumulate(model):
            with torch.no_grad():
                prepared = adapter.prepare_batch(pixel_values, args.context_length)
            loss, metrics = accelerator.unwrap_model(model).compute_loss(prepared, actions, global_step=completed_steps)
            accelerator.backward(loss)
            if accelerator.sync_gradients:
                accelerator.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optimizer.step()
            optimizer.zero_grad(set_to_none=True)

        if accelerator.sync_gradients:
            completed_steps += 1
            progress_bar.update(1)
            if completed_steps % args.log_steps == 0:
                logs = {key: float(value.detach().item()) for key, value in metrics.items()}
                logs['train/lr'] = optimizer.param_groups[0]['lr']
                if args.with_tracking:
                    accelerator.log(logs, step=completed_steps)
            if completed_steps % args.validation_steps == 0:
                evaluate(args, accelerator, adapter, model, eval_dataloader, evaluator, completed_steps)
            if completed_steps % args.checkpointing_steps == 0:
                if accelerator.is_main_process:
                    save_checkpoint(args.output_dir, accelerator, model, adapter, optimizer, completed_steps)
        if completed_steps >= args.max_train_steps:
            break

    if accelerator.is_main_process:
        final_dir = save_checkpoint(args.output_dir, accelerator, model, adapter, optimizer, completed_steps)
        logger.info(f'Final checkpoint saved to {final_dir}')


if __name__ == '__main__':
    main()
