import argparse
import json
import os
from pathlib import Path

import torch
from accelerate.utils import set_seed
from PIL import Image

from SAMPO.data import DATASET_NAMED_MIXES, SimpleRoboticDataLoaderv2
from SAMPO.sampo_plus import AdapterConfig, ContinuousLatentAdapter, parse_scales


def parse_args():
    parser = argparse.ArgumentParser(description='Validate and export a continuous latent adapter for SAMPO++.')
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--dataset_path', type=str, default='/data2/frame_datasets')
    parser.add_argument('--dataset_size', type=int, default=None)
    parser.add_argument('--oxe_data_mixes_type', type=str, default='select')
    parser.add_argument('--resolution', type=int, default=64)
    parser.add_argument('--context_length', type=int, default=2)
    parser.add_argument('--segment_length', type=int, default=16)
    parser.add_argument('--video_stepsize', type=int, default=1)
    parser.add_argument('--train_batch_size', type=int, default=4)
    parser.add_argument('--dataloader_num_workers', type=int, default=4)
    parser.add_argument('--sthsth_root_path', type=str, default='/data/something-something-v2/20bn-something-something-v2-frames-64')
    parser.add_argument('--vae_backend', type=str, default='debug', choices=['debug', 'lightningdit_vavae'])
    parser.add_argument('--vae_ckpt', type=str, default=None)
    parser.add_argument('--vae_repo_path', type=str, default=None)
    parser.add_argument('--latent_channels', type=int, default=32)
    parser.add_argument('--latent_scales', type=str, default='1,2,4,8,16')
    parser.add_argument('--model_resolution', type=int, default=256)
    parser.add_argument('--use_variational', action='store_true')
    return parser.parse_args()


def build_dataloader(args):
    augmentation_args = {
        'brightness': [0.9, 1.1],
        'contrast': [0.9, 1.1],
        'saturation': [0.9, 1.1],
        'hue': [-0.05, 0.05],
        'random_resized_crop_scale': (0.8, 1.0),
        'random_resized_crop_ratio': (0.9, 1.1),
        'no_aug': False,
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
    return SimpleRoboticDataLoaderv2(
        parent_dir=args.dataset_path,
        datasets=DATASET_NAMED_MIXES[args.oxe_data_mixes_type],
        batch_size=args.train_batch_size,
        num_workers=args.dataloader_num_workers,
        train=True,
        maxsize=args.dataset_size,
        image_size=args.resolution,
        sthsth_root_path=args.sthsth_root_path,
        load_action=False,
        **augmentation_args,
        **segment_args,
    )


def save_preview(frames: torch.Tensor, recon: torch.Tensor, output_path: Path):
    frames = frames[0].detach().cpu().clamp(0.0, 1.0)
    recon = recon[0].detach().cpu().clamp(0.0, 1.0)
    tiles = []
    for source, pred in zip(frames, recon):
        left = (source.permute(1, 2, 0).numpy() * 255).astype('uint8')
        right = (pred.permute(1, 2, 0).numpy() * 255).astype('uint8')
        pair = Image.fromarray(left)
        canvas = Image.new('RGB', (left.shape[1] * 2, left.shape[0]))
        canvas.paste(pair, (0, 0))
        canvas.paste(Image.fromarray(right), (left.shape[1], 0))
        tiles.append(canvas)
    merged = Image.new('RGB', (tiles[0].size[0], tiles[0].size[1] * len(tiles)))
    for idx, image in enumerate(tiles):
        merged.paste(image, (0, idx * image.size[1]))
    merged.save(output_path)


def main():
    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    set_seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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
    ).to(device).freeze()

    dataloader = build_dataloader(args)
    batch = next(iter(dataloader)).to(device, non_blocking=True)

    with torch.no_grad():
        prepared = adapter.prepare_batch(batch, args.context_length)
        recon = adapter.decode_frames(prepared['fine_latents'])

    metrics = {
        'recon_mse': torch.mean((batch - recon) ** 2).item(),
        'input_shape': list(batch.shape),
        'latent_shape': list(prepared['fine_latents'].shape),
        'latent_scales': list(parse_scales(args.latent_scales)),
        'value_range': [float(recon.min().item()), float(recon.max().item())],
    }

    adapter_dir = output_dir / 'adapter'
    adapter.save_pretrained(adapter_dir)
    with (output_dir / 'adapter_metrics.json').open('w', encoding='utf-8') as handle:
        json.dump(metrics, handle, indent=2)
    save_preview(batch, recon, output_dir / 'adapter_preview.png')

    print(json.dumps(metrics, indent=2))
    print(f'Saved adapter to {adapter_dir}')


if __name__ == '__main__':
    main()
