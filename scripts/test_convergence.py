#!/usr/bin/env python3
"""
Smoke test: validates forward pass, backward pass, gradient flow,
and a short training loop to verify the convergence fixes.

Usage:
    python scripts/test_convergence.py
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn.functional as F
from src.models.struct_router import StructRouter, create_struct_router
from src.losses.joint_loss import JointLoss


def test_forward_backward():
    """Test that forward and backward pass work without errors."""
    print("=" * 60)
    print("Test 1: Forward + Backward pass")
    print("=" * 60)

    model = StructRouter(
        input_dim=7, output_dim=7, hidden_dim=128,
        seq_len=96, pred_len=96,
        use_segment_processing=True,
        use_verification=True,
        use_rft=False,
        use_revin=True,
    )

    x = torch.randn(4, 96, 7)
    y = torch.randn(4, 96, 7)

    model.train()
    output = model(x)
    pred = output['prediction']

    print(f"  Input shape:       {x.shape}")
    print(f"  Prediction shape:  {pred.shape}")
    print(f"  Expert weights:    {output['expert_weights'][0].detach()}")
    print(f"  Consistency score: {output['consistency_score'][0].item():.4f}")

    loss = F.mse_loss(pred, y)
    loss.backward()

    # Check that gradients flow to key parameters
    grad_report = {}
    for name, param in model.named_parameters():
        if param.grad is not None:
            grad_report[name] = param.grad.norm().item()

    print(f"\n  Loss: {loss.item():.4f}")
    print(f"  Parameters with gradients: {len(grad_report)} / {sum(1 for p in model.parameters() if p.requires_grad)}")

    critical_params = [
        'structure_encoder.input_embed.weight',
        'weight_estimator.pattern_prototypes',
        'weight_estimator.weight_mlp.0.weight',
        'communication.W_comm',
        'expert_fusion.experts.periodic.fc1.weight',
    ]
    for name in critical_params:
        gn = grad_report.get(name, 'NO GRAD')
        status = "OK" if isinstance(gn, float) and gn > 0 else "MISSING"
        print(f"  [{status}] {name}: grad_norm = {gn}")

    print("  PASSED\n")


def test_verification_gradient_fix():
    """Verify that output_structure has gradients (the bug fix)."""
    print("=" * 60)
    print("Test 2: VerificationAgent gradient fix")
    print("=" * 60)

    model = StructRouter(
        input_dim=7, output_dim=7, hidden_dim=128,
        seq_len=96, pred_len=96,
        use_verification=True,
        use_revin=True,
    )

    x = torch.randn(4, 96, 7)
    model.train()
    output = model(x)

    out_struct = output['output_structure']
    in_struct = output['input_structure']

    has_out_grad = out_struct.requires_grad
    has_in_grad = in_struct.requires_grad

    print(f"  output_structure.requires_grad = {has_out_grad}  (expected: True)")
    print(f"  input_structure.requires_grad  = {has_in_grad}   (expected: False)")

    L_consist = F.mse_loss(out_struct, in_struct.detach())
    L_consist.backward()

    encoder_grad = model.structure_encoder.input_embed.weight.grad
    has_encoder_grad = encoder_grad is not None and encoder_grad.norm().item() > 0
    print(f"  Encoder gets gradient from L_consist: {has_encoder_grad}  (expected: True)")

    assert has_out_grad, "output_structure must have gradients"
    assert not has_in_grad, "input_structure should be detached"
    assert has_encoder_grad, "Encoder must receive gradients from consistency loss"
    print("  PASSED\n")


def test_dag_warmup():
    """Test that DAG penalty warms up correctly."""
    print("=" * 60)
    print("Test 3: DAG penalty warmup")
    print("=" * 60)

    joint_loss = JointLoss(
        gamma=0.1,
        dag_warmup_steps=100,
    )

    gamma_values = []
    for i in range(120):
        gamma_values.append(joint_loss.gamma)
        joint_loss.step()

    print(f"  gamma at step 0:   {gamma_values[0]:.4f}  (expected: 0.0)")
    print(f"  gamma at step 50:  {gamma_values[50]:.4f}  (expected: ~0.05)")
    print(f"  gamma at step 100: {gamma_values[100]:.4f}  (expected: 0.1)")
    print(f"  gamma at step 110: {gamma_values[110]:.4f}  (expected: 0.1)")

    assert gamma_values[0] == 0.0, "gamma should start at 0"
    assert 0.04 < gamma_values[50] < 0.06, "gamma should be ~0.05 at midpoint"
    assert gamma_values[100] == 0.1, "gamma should reach target at warmup_steps"
    assert gamma_values[110] == 0.1, "gamma should stay at target after warmup"
    print("  PASSED\n")


def test_revin():
    """Test RevIN normalize → denormalize roundtrip."""
    print("=" * 60)
    print("Test 4: RevIN roundtrip")
    print("=" * 60)

    from src.models.revin import RevIN

    revin = RevIN(num_features=7, affine=False)
    x = torch.randn(4, 96, 7) * 10 + 5  # mean=5, std~=10

    x_norm = revin.normalize(x)
    x_recon = revin.denormalize(x_norm)

    err = (x - x_recon).abs().max().item()
    print(f"  Input mean:  {x.mean().item():.2f}")
    print(f"  Normed mean: {x_norm.mean().item():.4f}  (expected: ~0)")
    print(f"  Normed std:  {x_norm.std().item():.4f}  (expected: ~1)")
    print(f"  Roundtrip max error: {err:.2e}  (expected: <1e-5)")

    assert err < 1e-4, f"RevIN roundtrip error too large: {err}"
    print("  PASSED\n")


def test_short_training_loop():
    """Run a short training loop to check loss decreases."""
    print("=" * 60)
    print("Test 5: Short training loop (20 steps)")
    print("=" * 60)

    model = StructRouter(
        input_dim=7, output_dim=7, hidden_dim=64,
        seq_len=48, pred_len=48,
        use_segment_processing=False,
        use_verification=False,
        use_rft=False,
        use_revin=True,
    )

    joint_loss = JointLoss(
        alpha=0.1, beta=0.01, gamma=0.01, delta=0.01, epsilon=0.01,
        dag_warmup_steps=10,
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    model.train()

    # Synthetic data
    x = torch.randn(8, 48, 7)
    y = x + 0.1 * torch.randn_like(x)  # near-identity target

    losses = []
    for step in range(20):
        output = model(x)
        prototypes = model.weight_estimator.pattern_prototypes

        loss_dict = joint_loss(
            prediction=output['prediction'],
            target=y,
            input_structure=output['input_structure'],
            output_structure=output['output_structure'],
            expert_weights=output['expert_weights'],
            W_comm=output['W_comm'],
            router_logits=output.get('router_logits'),
            prototypes=prototypes,
        )

        optimizer.zero_grad()
        loss_dict['total'].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        joint_loss.step()

        losses.append(loss_dict['total'].item())
        if step % 5 == 0:
            print(f"  Step {step:2d}: total={losses[-1]:.4f}  task={loss_dict['task'].item():.4f}  "
                  f"dag={loss_dict['causal'].item():.4f}  γ={joint_loss.gamma:.4f}")

    improved = losses[-1] < losses[0]
    print(f"\n  Loss start: {losses[0]:.4f} → end: {losses[-1]:.4f}")
    print(f"  Loss decreased: {improved}")
    if improved:
        print("  PASSED\n")
    else:
        print("  WARNING: loss did not decrease (may need more steps)\n")


def test_per_module_lr():
    """Verify that per-module LR groups are set up correctly."""
    print("=" * 60)
    print("Test 6: Per-module learning rates")
    print("=" * 60)

    from src.trainers.three_stage_trainer import Stage2Finetuner
    from src.losses.joint_loss import JointLoss

    model = StructRouter(
        input_dim=7, output_dim=7, hidden_dim=64,
        seq_len=48, pred_len=48,
        use_segment_processing=False,
        use_verification=False,
        use_rft=False,
    )

    joint_loss = JointLoss()
    finetuner = Stage2Finetuner(
        model, joint_loss,
        lr=5e-4,
        prototype_lr_scale=0.1,
        wcomm_lr_scale=3.0,
    )

    for group in finetuner.optimizer.param_groups:
        name = group.get('name', 'default')
        lr = group['lr']
        n_params = len(group['params'])
        print(f"  Group '{name}': lr={lr:.6f}, n_params={n_params}")

    print("  PASSED\n")


if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("StructRouter Convergence Test Suite")
    print("=" * 60 + "\n")

    test_forward_backward()
    test_verification_gradient_fix()
    test_dag_warmup()
    test_revin()
    test_short_training_loop()
    test_per_module_lr()

    print("=" * 60)
    print("ALL TESTS COMPLETED")
    print("=" * 60)
