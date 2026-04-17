"""
SOAR core algorithm: stochastic rollout and auxiliary supervision.

Given a noisy sample z_t0 at time t0, SOAR constructs auxiliary supervision
points via one-step rollout followed by interpolation:
  1. Take one inference step: t1 = max(t0 - 1/num_sampling_steps, 0)
  2. Build 1 ODE path and (M-1) stochastic paths to z_t1'
  3. Interpolate N points between z_t1' and z_1 (pure noise)
  4. Supervise the model at these interpolated points with corrected targets
"""

import math

import torch


def t_to_sigma_timestep(t, noise_scheduler):
    """Convert normalized t in [0, 1] to scheduler sigma and timestep."""
    num_steps = len(noise_scheduler.timesteps)
    indices = ((1.0 - t) * num_steps).long().clamp(0, num_steps - 1)
    sigmas = noise_scheduler.sigmas.to(device=t.device, dtype=t.dtype)[indices]
    timesteps = noise_scheduler.timesteps.to(device=t.device)[indices]
    return sigmas, timesteps


def sigma_to_t(sigma, noise_scheduler):
    """Convert sigma back to normalized t in [0, 1]."""
    num_steps = len(noise_scheduler.timesteps)
    sched_sigmas = noise_scheduler.sigmas[:num_steps].to(
        device=sigma.device, dtype=sigma.dtype
    )
    indices = torch.searchsorted(-sched_sigmas, -sigma).clamp(0, num_steps - 1)
    return 1.0 - indices.float() / num_steps


def stochastic_rollout_step(
    sample,
    velocity,
    sigma_curr,
    sigma_next,
    sde_rollout_type,
    sde_noise_scale,
    sigma_max_value,
):
    """Apply one stochastic rollout step.

    Supported modes:
      - simple:   additive Euler-Maruyama noise
      - sde:      diffusers-style stochastic sampling
      - flow_sde: reverse-SDE step (flow_grpo)
      - cps:      Coefficients-Preserving Sampling
    """
    sample_f32 = sample.float()
    velocity_f32 = velocity.float()
    sigma_curr_4d = sigma_curr.view(-1, 1, 1, 1).float()
    sigma_next_4d = sigma_next.view(-1, 1, 1, 1).float()
    dt = sigma_next_4d - sigma_curr_4d

    if sde_rollout_type == "simple":
        next_sample = sample_f32 + velocity_f32 * dt
        next_sample = next_sample + sde_noise_scale * torch.sqrt(
            dt.abs()
        ) * torch.randn_like(sample_f32)

    elif sde_rollout_type == "sde":
        pred_original_sample = sample_f32 - sigma_curr_4d * velocity_f32
        next_sample = (1.0 - sigma_next_4d) * pred_original_sample + sigma_next_4d * torch.randn_like(sample_f32)

    elif sde_rollout_type == "flow_sde":
        sigma_curr_safe = torch.where(
            sigma_curr_4d == 1,
            torch.full_like(sigma_curr_4d, sigma_max_value),
            sigma_curr_4d,
        )
        std_dev_t = (
            torch.sqrt(sigma_curr_4d / (1 - sigma_curr_safe)) * sde_noise_scale
        )
        prev_sample_mean = sample_f32 * (
            1 + (std_dev_t.square() / (2 * sigma_curr_4d)) * dt
        )
        prev_sample_mean = prev_sample_mean + velocity_f32 * (
            1 + (std_dev_t.square() * (1 - sigma_curr_4d) / (2 * sigma_curr_4d))
        ) * dt
        next_sample = prev_sample_mean + std_dev_t * torch.sqrt(
            (-dt).clamp_min(0.0)
        ) * torch.randn_like(sample_f32)

    elif sde_rollout_type == "cps":
        std_dev_t = sigma_next_4d * math.sin(sde_noise_scale * math.pi / 2)
        pred_original_sample = sample_f32 - sigma_curr_4d * velocity_f32
        noise_estimate = sample_f32 + velocity_f32 * (1 - sigma_curr_4d)
        prev_sample_mean = pred_original_sample * (1 - sigma_next_4d)
        prev_sample_mean = prev_sample_mean + noise_estimate * torch.sqrt(
            (sigma_next_4d.square() - std_dev_t.square()).clamp_min(0.0)
        )
        next_sample = prev_sample_mean + std_dev_t * torch.randn_like(sample_f32)

    else:
        raise ValueError(f"Unsupported sde_rollout_type: {sde_rollout_type}")

    return next_sample.to(dtype=sample.dtype)


@torch.no_grad()
def single_step_aux_points(
    z_t0,
    t0,
    v_cfg,
    z_1,
    num_paths,
    points_per_path,
    sigma_upper_ratio,
    noise_scheduler,
    num_sampling_steps,
    sde_rollout_type="cps",
    sde_noise_scale=0.5,
):
    """Construct auxiliary supervision points via one-step rollout + interpolation.

    1. From z_t0, take one ODE step to t1 = max(t0 - 1/num_sampling_steps, 0).
    2. Path 0 uses ODE; paths 1..M-1 use stochastic updates -> z_t1'.
    3. For each path, interpolate `points_per_path` points between z_t1' and z_1
       in the sigma range [sigma_t1, min(sigma_upper_ratio * sigma_t0, 1)].
    4. Samples hitting the t=0 boundary only keep the single ODE path.

    Returns a list of dicts, each with keys:
        sample_indices, latents, sigmas, timesteps
    """
    if num_paths < 1:
        return []

    device = z_t0.device
    dtype = z_t0.dtype
    batch_size = z_t0.shape[0]

    sigma_t0_1d, _ = t_to_sigma_timestep(t0, noise_scheduler)
    sigma_t0_1d = sigma_t0_1d.detach()
    sigma_t0 = sigma_t0_1d.view(-1, 1, 1, 1).to(dtype=dtype)

    t1 = (t0.detach() - 1.0 / float(num_sampling_steps)).clamp_min(0.0)
    sigma_t1_1d, timestep_t1 = t_to_sigma_timestep(t1, noise_scheduler)
    sigma_t1_1d = sigma_t1_1d.detach()
    sigma_t1 = sigma_t1_1d.view(-1, 1, 1, 1).to(dtype=dtype)
    timestep_t1 = timestep_t1.detach()

    sigma_upper = torch.ones_like(sigma_t0_1d)
    hit_clean_boundary = t1 <= 0

    train_sigmas = noise_scheduler.sigmas.to(device=device, dtype=dtype)
    sigma_max_value = (
        float(train_sigmas[1].detach().cpu())
        if train_sigmas.shape[0] > 1
        else float(train_sigmas[0].detach().cpu())
    )

    points = []
    full_indices = torch.arange(batch_size, device=device, dtype=torch.long)

    def _append_interpolated(path_indices, z_t1_prime, sigma_t1_sub, sigma_upper_sub):
        if path_indices.numel() == 0:
            return
        rand_fracs = torch.rand(
            points_per_path, path_indices.shape[0],
            device=device, dtype=sigma_t1_sub.dtype,
        )
        sigma_targets = sigma_t1_sub.unsqueeze(0) + rand_fracs * (
            sigma_upper_sub - sigma_t1_sub
        ).unsqueeze(0)

        for pidx in range(points_per_path):
            sigma_t_prime_1d = sigma_targets[pidx]
            interp_denom = (1.0 - sigma_t1_sub).clamp_min(1e-8)
            alpha = ((sigma_t_prime_1d - sigma_t1_sub) / interp_denom).clamp(0.0, 1.0)
            alpha_4d = alpha.view(-1, 1, 1, 1).to(dtype=dtype)
            z_interp = (
                (1.0 - alpha_4d) * z_t1_prime + alpha_4d * z_1[path_indices]
            ).detach()
            t_t_prime = sigma_to_t(sigma_t_prime_1d, noise_scheduler)
            _, ts_t_prime = t_to_sigma_timestep(t_t_prime, noise_scheduler)
            points.append({
                "sample_indices": path_indices.detach(),
                "latents": z_interp,
                "sigmas": sigma_t_prime_1d.detach(),
                "timesteps": ts_t_prime.detach(),
            })

    # ODE path (always present)
    dt_one = sigma_t1 - sigma_t0
    z_t1_prime_ode = (z_t0 + v_cfg * dt_one).detach()
    _append_interpolated(full_indices, z_t1_prime_ode, sigma_t1_1d, sigma_upper)

    # Stochastic paths
    if num_paths > 1:
        active_sde = (~hit_clean_boundary).nonzero(as_tuple=False).squeeze(1)
        if active_sde.numel() > 0:
            z_t0_sde = z_t0[active_sde]
            v_cfg_sde = v_cfg[active_sde]
            sigma_t0_sde = sigma_t0_1d[active_sde]
            sigma_t1_sde = sigma_t1_1d[active_sde]
            sigma_upper_sde = sigma_upper[active_sde]

            for _ in range(num_paths - 1):
                z_t1_prime_sde = stochastic_rollout_step(
                    sample=z_t0_sde,
                    velocity=v_cfg_sde,
                    sigma_curr=sigma_t0_sde,
                    sigma_next=sigma_t1_sde,
                    sde_rollout_type=sde_rollout_type,
                    sde_noise_scale=sde_noise_scale,
                    sigma_max_value=sigma_max_value,
                ).to(dtype=dtype).detach()

                _append_interpolated(
                    active_sde, z_t1_prime_sde, sigma_t1_sde, sigma_upper_sde
                )

    return points
