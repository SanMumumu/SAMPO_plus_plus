import argparse
import os
import random
from pathlib import Path

import imageio
import numpy as np
import torch

from SAMPO.sampo_plus import ContinuousLatentAdapter, SampoPlusModel
from utils import NPZParser


def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_args():
    parser = argparse.ArgumentParser(description='Run SAMPO++ continuous rollout inference.')
    parser.add_argument('--pretrained_model_name_or_path', type=str, required=True, help='checkpoint directory containing adapter/ and world_model/')
    parser.add_argument('--input_path', type=str, required=True, help='path to input npz file')
    parser.add_argument('--dataset_name', type=str, required=True, help='dataset name')
    parser.add_argument('--output_path', type=str, default='outputs', help='path to save predicted videos')
    parser.add_argument('--context_length', type=int, default=2, help='number of context frames')
    parser.add_argument('--segment_length', type=int, default=16, help='total rollout length including context')
    parser.add_argument('--resolution', type=int, default=64, help='frame resolution')
    parser.add_argument('--action_conditioned', action='store_true', help='enable action-conditioned rollout')
    parser.add_argument('--repeat_times', type=int, default=1, help='number of stochastic rollouts')
    parser.add_argument('--num_flow_steps', type=int, default=25, help='number of Euler steps per scale')
    parser.add_argument('--use_rectification', action='store_true')
    parser.add_argument('--seed', type=int, default=0)
    return parser.parse_args()


@torch.no_grad()
def predict(args, adapter, model, inputs, actions=None):
    device = next(model.parameters()).device
    pixel_values = inputs.to(device, non_blocking=True).unsqueeze(0)
    actions = actions.to(device, non_blocking=True).unsqueeze(0) if actions is not None else None
    context_frames = pixel_values[:, : args.context_length]
    output_dir = Path(args.output_path)
    output_dir.mkdir(parents=True, exist_ok=True)

    for sample_index in range(args.repeat_times):
        rollout = model.rollout(
            adapter,
            context_frames=context_frames,
            actions=actions,
            num_future_frames=args.segment_length - args.context_length,
            num_flow_steps=args.num_flow_steps,
            use_rectification=args.use_rectification,
        )
        pred = rollout['frames'][0].detach().cpu().clamp(0.0, 1.0)
        gt = pixel_values[0].detach().cpu().clamp(0.0, 1.0)
        frames = []
        for gt_frame, pred_frame in zip(gt, pred):
            left = (gt_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            right = (pred_frame.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            frames.append(np.concatenate([left, right], axis=1))
        imageio.mimsave(output_dir / f'pred-samples-{sample_index}.gif', frames, fps=4, loop=0)


def main():
    args = parse_args()
    set_seed(args.seed)
    checkpoint_dir = Path(args.pretrained_model_name_or_path)
    adapter = ContinuousLatentAdapter.from_pretrained(checkpoint_dir / 'adapter')
    model = SampoPlusModel.from_pretrained(checkpoint_dir / 'world_model')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    adapter = adapter.to(device).freeze()
    model = model.to(device).eval()

    npz_parser = NPZParser(args.segment_length, args.resolution)
    inputs, actions = npz_parser.parse(args.input_path, args.dataset_name, load_action=args.action_conditioned)
    predict(args, adapter, model, inputs, actions)


if __name__ == '__main__':
    main()
