import torch

from SAMPO.sampo_plus import (
    AdapterConfig,
    build_sampo_plus,
    ContinuousLatentAdapter,
    FlowRendererConfig,
    TemporalPlanner,
    TemporalPlannerConfig,
)
from SAMPO.sampo_plus.renderer import SILoss, ScaleAwareFlowRenderer


def make_adapter() -> ContinuousLatentAdapter:
    return ContinuousLatentAdapter(
        AdapterConfig(
            backend='debug',
            input_resolution=64,
            model_resolution=256,
            latent_resolution=16,
            latent_channels=16,
            latent_scales=(1, 2, 4, 8, 16),
        )
    )


def make_video(batch_size: int = 2, steps: int = 6) -> torch.Tensor:
    torch.manual_seed(0)
    return torch.rand(batch_size, steps, 3, 64, 64)


def make_actions(batch_size: int = 2, steps: int = 6, action_dim: int = 4) -> torch.Tensor:
    torch.manual_seed(1)
    return torch.randn(batch_size, steps, action_dim)


def test_adapter_roundtrip_and_pyramid_shapes():
    adapter = make_adapter()
    video = make_video()

    prepared = adapter.prepare_batch(video, context_length=2)
    recon = adapter.decode_frames(prepared['fine_latents'])

    assert prepared['fine_latents'].shape == (2, 6, 16, 16, 16)
    assert recon.shape == video.shape
    assert recon.dtype == video.dtype
    assert 0.0 <= float(recon.min()) <= 1.0
    assert 0.0 <= float(recon.max()) <= 1.0
    assert prepared['pyramid_1'].shape == (2, 6, 16, 1, 1)
    assert prepared['pyramid_16'].shape == (2, 6, 16, 16, 16)


def test_planner_teacher_forcing_matches_step_shape_contract():
    adapter = make_adapter()
    planner = TemporalPlanner(
        TemporalPlannerConfig(
            latent_channels=adapter.latent_channels,
            hidden_size=128,
            plan_size=96,
            action_dim=4,
            num_layers=2,
            num_heads=4,
            max_frames=16,
        )
    )
    video = make_video()
    actions = make_actions()
    fine_latents = adapter.encode_frames(video)

    plans = planner.forward_teacher_forcing(fine_latents, actions, context_length=2)
    step_plan, cache = planner.step(fine_latents[:, :2], actions[:, :2], cache=None)
    step_plan_cached, _ = planner.step(fine_latents[:, :2], actions[:, :2], cache=cache)

    assert plans.shape == (2, 4, 96)
    assert step_plan.shape == (2, 96)
    assert step_plan_cached.shape == (2, 96)
    assert torch.allclose(step_plan, step_plan_cached)


def test_renderer_si_loss_is_finite_and_backwardable():
    torch.manual_seed(2)
    renderer = ScaleAwareFlowRenderer(
        FlowRendererConfig(
            latent_channels=16,
            plan_size=96,
            action_dim=4,
            hidden_size=128,
            num_layers=2,
            num_heads=4,
            max_frames=16,
            max_scales=8,
        )
    )
    target = torch.randn(2, 16, 8, 8)
    coarse = torch.randn(2, 16, 4, 4)
    plan = torch.randn(2, 96)
    action = torch.randn(2, 4)

    loss = SILoss()(renderer, target, coarse, plan, action, frame_index=0, scale_index=2).mean()
    loss.backward()

    assert torch.isfinite(loss)
    grads = [param.grad for param in renderer.parameters() if param.grad is not None]
    assert grads
    assert all(torch.isfinite(grad).all() for grad in grads)


def test_world_model_loss_and_rollout_shapes():
    adapter = make_adapter().freeze()
    model = build_sampo_plus(
        adapter,
        context_length=2,
        latent_scales=(1, 2, 4, 8, 16),
        flow_loss_weights=(1.0, 1.0, 1.0, 1.0, 1.0),
        action_dim=4,
        plan_size=96,
        planner_hidden_size=128,
        planner_num_layers=2,
        planner_num_heads=4,
        renderer_hidden_size=128,
        renderer_num_layers=2,
        renderer_num_heads=4,
        max_frames=16,
    )
    video = make_video()
    actions = make_actions()
    prepared = adapter.prepare_batch(video, context_length=2)

    loss, metrics = model.compute_loss(prepared, actions)
    rollout = model.rollout(
        adapter,
        context_frames=video[:, :2],
        actions=actions,
        num_future_frames=4,
        num_flow_steps=4,
        use_rectification=False,
    )

    assert torch.isfinite(loss)
    assert 'loss' in metrics
    assert rollout['frames'].shape == video.shape
    assert rollout['predicted_fine'].shape == (2, 4, 16, 16, 16)
    assert rollout['plans'].shape == (2, 4, 96)
