"""Shape / property contracts for the refactored ``sampo_pp`` package.

Run from the package root:  ``pytest tests/test_sampo_pp.py -q``

Covers the legacy contracts plus the three TPAMI-revision contributions:
multi-scale planner, ACVF (no-op + counterfactual), and PCD-RoPE mode routing.
"""
import torch

from sampo_pp import (
    ACVFBlock,
    AdapterConfig,
    ContinuousLatentAdapter,
    FlowRendererConfig,
    Pi0ActionSpec,
    Pi0SampoInterface,
    ScaleAwareFlowRenderer,
    SILoss,
    TemporalPlanner,
    TemporalPlannerConfig,
    build_sampo_plus,
    build_spatial_rope,
)
from sampo_pp.metrics import (
    action_alignment,
    action_energy_profile,
    counterfactual_accuracy,
    noop_stability,
    object_permanence_proxy,
    rollout_drift,
)

SCALES = (1, 2, 4, 8, 16)


def make_adapter() -> ContinuousLatentAdapter:
    return ContinuousLatentAdapter(
        AdapterConfig(backend='debug', input_resolution=64, model_resolution=256,
                      latent_resolution=16, latent_channels=16, latent_scales=SCALES)
    )


def make_video(batch_size=2, steps=6):
    torch.manual_seed(0)
    return torch.rand(batch_size, steps, 3, 64, 64)


def make_actions(batch_size=2, steps=6, action_dim=4):
    torch.manual_seed(1)
    return torch.randn(batch_size, steps, action_dim)


# --------------------------------------------------------------------------- #
# Adapter
# --------------------------------------------------------------------------- #
def test_adapter_roundtrip_and_pyramid_shapes():
    adapter = make_adapter()
    video = make_video()
    prepared = adapter.prepare_batch(video, context_length=2)
    recon = adapter.decode_frames(prepared['fine_latents'])

    assert prepared['fine_latents'].shape == (2, 6, 16, 16, 16)
    assert recon.shape == video.shape
    assert recon.dtype == video.dtype
    assert 0.0 <= float(recon.min()) and float(recon.max()) <= 1.0
    assert prepared['pyramid_1'].shape == (2, 6, 16, 1, 1)
    assert prepared['pyramid_16'].shape == (2, 6, 16, 16, 16)


# --------------------------------------------------------------------------- #
# Planner (single- and multi-scale)
# --------------------------------------------------------------------------- #
def _planner(multi_scale):
    return TemporalPlanner(TemporalPlannerConfig(
        latent_channels=16, hidden_size=128, plan_size=96, action_dim=4,
        num_layers=2, num_heads=4, max_frames=16, latent_scales=SCALES, multi_scale=multi_scale,
    ))


def test_planner_single_scale_contract():
    adapter = make_adapter()
    planner = _planner(multi_scale=False)
    fine = adapter.encode_frames(make_video())
    actions = make_actions()

    plans = planner.forward_teacher_forcing(fine, actions, context_length=2)
    step_plan, cache = planner.step(fine[:, :2], actions[:, :2], cache=None)
    step_cached, _ = planner.step(fine[:, :2], actions[:, :2], cache=cache)

    assert plans.shape == (2, 4, 96)
    assert step_plan.shape == (2, 96)
    assert torch.allclose(step_plan, step_cached)


def test_planner_multi_scale_contract():
    adapter = make_adapter()
    planner = _planner(multi_scale=True)
    fine = adapter.encode_frames(make_video())
    actions = make_actions()

    plans = planner.forward_teacher_forcing(fine, actions, context_length=2)
    step_plan, cache = planner.step(fine[:, :2], actions[:, :2], cache=None)
    step_cached, _ = planner.step(fine[:, :2], actions[:, :2], cache=cache)

    assert plans.shape == (2, 4, len(SCALES), 96)
    assert step_plan.shape == (2, len(SCALES), 96)
    assert torch.allclose(step_plan, step_cached)


# --------------------------------------------------------------------------- #
# Renderer + SILoss
# --------------------------------------------------------------------------- #
def _renderer(action_mode='acvf', rope_mode='pcd'):
    return ScaleAwareFlowRenderer(FlowRendererConfig(
        latent_channels=16, plan_size=96, action_dim=4, hidden_size=128,
        num_layers=2, num_heads=4, max_frames=16, max_scales=8,
        action_mode=action_mode, rope_mode=rope_mode,
    ))


def test_renderer_si_loss_is_finite_and_backwardable():
    torch.manual_seed(2)
    renderer = _renderer()
    target, coarse = torch.randn(2, 16, 8, 8), torch.randn(2, 16, 4, 4)
    plan, action = torch.randn(2, 96), torch.randn(2, 4)

    loss = SILoss()(renderer, target, coarse, plan, action, frame_index=0, scale_index=2).mean()
    loss.backward()

    assert torch.isfinite(loss)
    grads = [p.grad for p in renderer.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rope_modes_all_run():
    for mode in ('learned', 'rope2d', 'spacetime', 'sarope4d', 'pcd'):
        renderer = _renderer(rope_mode=mode)
        x = torch.randn(2, 16, 8, 8)
        v = renderer(x, torch.rand(2), None, torch.randn(2, 96), torch.randn(2, 4), 0, 2)
        assert v.shape == (2, 16, 8, 8)


def test_action_mode_routing_runs():
    for mode in ('concat', 'crossattn', 'adaln', 'adm', 'acvf'):
        renderer = _renderer(action_mode=mode)
        x = torch.randn(2, 16, 8, 8)
        v = renderer(x, torch.rand(2), None, torch.randn(2, 96), torch.randn(2, 4), 0, 2)
        assert v.shape == (2, 16, 8, 8)


# --------------------------------------------------------------------------- #
# ACVF properties
# --------------------------------------------------------------------------- #
def test_acvf_is_noop_safe_and_zero_init():
    torch.manual_seed(3)
    block = ACVFBlock(dim=32, action_dim=4, num_scales=8)
    u = torch.randn(2, 9, 32)
    h_s = torch.randn(2, 32)

    # zero-init: residual is exactly 0 for ANY action at initialization
    r_init = block.control_residual(u, h_s, torch.randn(2, 4), s=2)
    assert r_init.abs().max() < 1e-6

    # no-op safety holds even after the weights are perturbed (psi/proj have no bias)
    for p in block.parameters():
        p.data = torch.randn_like(p)
    r_noop = block.control_residual(u, h_s, torch.zeros(2, 4), s=2)
    assert r_noop.abs().max() < 1e-5
    # a non-zero action now produces a non-trivial control signal
    r_act = block.control_residual(u, h_s, torch.randn(2, 4), s=2)
    assert r_act.abs().max() > 1e-4


def test_acvf_counterfactual_toy_fit_separates_true_from_wrong():
    torch.manual_seed(4)
    B = 6
    renderer = _renderer()
    opt = torch.optim.Adam(renderer.parameters(), lr=2e-3)
    plan = torch.randn(B, 96)
    action = torch.randn(B, 4)
    # strong, per-sample action-dependent target so the action genuinely carries
    # signal: a DC field proportional to action[:, 0] over light texture.
    base = 0.1 * torch.randn(B, 16, 8, 8)
    target = base + 3.0 * action[:, 0].view(B, 1, 1, 1)

    loss_fn = SILoss()
    for _ in range(150):
        opt.zero_grad()
        loss_fn(renderer, target, None, plan, action, 0, 2).mean().backward()
        opt.step()

    # average over many flow draws to remove single-sample noise
    wrong = torch.roll(action, shifts=1, dims=0)
    d_true, d_wrong = 0.0, 0.0
    with torch.no_grad():
        for _ in range(32):
            t = torch.rand(B)
            x0 = torch.randn_like(target)
            x_t = (1 - t.view(-1, 1, 1, 1)) * x0 + t.view(-1, 1, 1, 1) * target
            tv = target - x0
            d_true += (renderer(x_t, t, None, plan, action, 0, 2) - tv).pow(2).mean()
            d_wrong += (renderer(x_t, t, None, plan, wrong, 0, 2) - tv).pow(2).mean()
    assert d_true < d_wrong


# --------------------------------------------------------------------------- #
# World model end-to-end
# --------------------------------------------------------------------------- #
def _model(multi_scale=True, **kw):
    adapter = make_adapter().freeze()
    model = build_sampo_plus(
        adapter, context_length=2, latent_scales=SCALES, action_dim=4, plan_size=96,
        planner_hidden_size=128, planner_num_layers=2, planner_num_heads=4,
        renderer_hidden_size=128, renderer_num_layers=2, renderer_num_heads=4,
        max_frames=16, multi_scale=multi_scale, **kw,
    )
    return adapter, model


def test_world_model_loss_and_rollout_multi_scale():
    adapter, model = _model(multi_scale=True)
    video, actions = make_video(), make_actions()
    prepared = adapter.prepare_batch(video, context_length=2)

    loss, metrics = model.compute_loss(prepared, actions)
    rollout = model.rollout(adapter, context_frames=video[:, :2], actions=actions,
                            num_future_frames=4, num_flow_steps=4, use_rectification=False)

    assert torch.isfinite(loss)
    assert {'loss', 'loss_flow', 'loss_noop', 'loss_cf'} <= set(metrics)
    assert rollout['frames'].shape == video.shape
    assert rollout['predicted_fine'].shape == (2, 4, 16, 16, 16)
    assert rollout['plans'].shape == (2, 4, len(SCALES), 96)


def test_world_model_single_scale_rollout_contract():
    adapter, model = _model(multi_scale=False)
    video, actions = make_video(), make_actions()
    rollout = model.rollout(adapter, context_frames=video[:, :2], actions=actions,
                            num_future_frames=4, num_flow_steps=4)
    assert rollout['plans'].shape == (2, 4, 96)
    assert rollout['frames'].shape == video.shape


def test_pi0_interface_builds_context_padded_rollout_actions():
    adapter, model = _model(multi_scale=True)
    video = make_video(batch_size=1, steps=6)
    pi0_actions = torch.randn(3, 4)
    interface = Pi0SampoInterface(model, adapter, Pi0ActionSpec(action_dim=4, horizon=4))

    actions = interface.build_rollout_actions(video[:, :2], pi0_actions)
    rollout = interface.rollout(video[:, :2], pi0_actions, num_future_frames=4, num_flow_steps=2)

    assert actions.shape == (1, 6, 4)
    assert float(actions[:, :2].abs().max()) == 0.0
    assert float(actions[:, -1].abs().max()) == 0.0
    assert rollout['frames'].shape == video.shape


def test_long_euler_rollout_stays_finite():
    renderer = _renderer()
    plan, action = torch.randn(2, 96), torch.randn(2, 4)
    x = torch.randn(2, 16, 8, 8)
    with torch.no_grad():
        dt = 1.0 / 100
        for step in range(100):
            v = renderer(x, torch.full((2,), step * dt), None, plan, action, 0, 2)
            x = x + dt * v
    assert torch.isfinite(x).all()


# --------------------------------------------------------------------------- #
# World-model-native metrics
# --------------------------------------------------------------------------- #
def test_metrics_smoke():
    adapter, model = _model(multi_scale=True)
    video, actions = make_video(), make_actions()
    prepared = adapter.prepare_batch(video, context_length=2)

    drift = rollout_drift(torch.randn(2, 5, 16, 8, 8), torch.randn(2, 5, 16, 8, 8))
    assert drift['per_step'].shape == (5,) and torch.isfinite(drift['slope'])

    acc = counterfactual_accuracy(model, prepared, actions)
    assert 0.0 <= float(acc) <= 1.0

    stab = noop_stability(model, adapter, video[:, :2], num_future_frames=3, num_flow_steps=3)
    assert torch.isfinite(stab)

    energy = action_energy_profile(model, prepared, actions)
    assert energy.shape == (len(SCALES),) and torch.isfinite(energy).all()

    align = action_alignment(model, prepared, actions)
    assert torch.isfinite(align) and -1.0 <= float(align) <= 1.0

    perm = object_permanence_proxy(torch.randn(2, 5, 16, 8, 8), torch.randn(2, 5, 16, 8, 8))
    assert torch.isfinite(perm)


# --------------------------------------------------------------------------- #
# Rollout-aware training (Phase 5)
# --------------------------------------------------------------------------- #
def test_rollout_aware_adds_consistency_loss():
    adapter, model = _model(multi_scale=True, rollout_aware_p=1.0, rollout_k=2)
    video, actions = make_video(), make_actions()
    prepared = adapter.prepare_batch(video, context_length=2)

    loss, metrics = model.compute_loss(prepared, actions, global_step=100)
    assert 'loss_rollout' in metrics
    assert torch.isfinite(loss) and float(metrics['loss_rollout']) > 0.0
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads)


def test_rollout_aware_off_by_default():
    adapter, model = _model(multi_scale=True)
    video, actions = make_video(), make_actions()
    prepared = adapter.prepare_batch(video, context_length=2)
    _, metrics = model.compute_loss(prepared, actions, global_step=100)
    assert float(metrics['loss_rollout']) == 0.0
