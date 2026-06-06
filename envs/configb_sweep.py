"""
configb_sweep.py — Hyperparameter sweep for Config B.

Two-stage protocol:
  Stage 1: Train each candidate config for `sweep_timesteps` (default 100k)
           and evaluate deterministically on 500 episodes of the clinical env.
           Result: ranking of configs per algorithm.
  Stage 2: Take the best config and retrain at `final_timesteps` (default 500k)
           for the report's main results.

Each candidate config is a delta over a sensible baseline. The baseline
config matches what we ran in our first failed attempt — it stays in the
sweep so the report can show how much tuning actually helped.

Usage from the notebook
-----------------------
    from envs.configb_sweep import (
        DQN_CONFIGS, PPO_CONFIGS,
        run_sweep, train_best,
    )

    results_dqn = run_sweep('dqn', DQN_CONFIGS, total_timesteps=100_000)
    results_ppo = run_sweep('ppo', PPO_CONFIGS, total_timesteps=100_000)

    # Pick the best by survival rate and train it for real
    best_dqn = results_dqn.sort_values('survival', ascending=False).iloc[0]['config']
    best_ppo = results_ppo.sort_values('survival', ascending=False).iloc[0]['config']

    model_dqn, log_dqn = train_best('dqn', best_dqn, total_timesteps=500_000)
    model_ppo, log_ppo = train_best('ppo', best_ppo, total_timesteps=500_000)
"""

from __future__ import annotations

import os
import time
from typing import Dict

import numpy as np
import pandas as pd

from envs.configb import (
    make_train_env, evaluate_agent,
    LOG_ROOT, MODEL_ROOT,
)



# 1. Candidate configurations
def _dqn_cfg(**overrides) -> dict:
    """DQN baseline kwargs + selective overrides."""
    base = dict(
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,
        gamma=1.0,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    base.update(overrides)
    return base


def _ppo_cfg(**overrides) -> dict:
    """PPO baseline kwargs + selective overrides."""
    base = dict(
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=1.0,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
    )
    base.update(overrides)
    return base


DQN_CONFIGS: Dict[str, dict] = {
    # The original config that failed — kept as a documented negative baseline
    'baseline':      _dqn_cfg(),
    'lr_higher':     _dqn_cfg(learning_rate=3e-4),
    'lr_max':        _dqn_cfg(learning_rate=1e-3),
    'bigger_net':    _dqn_cfg(learning_rate=3e-4, policy_kwargs=dict(net_arch=[128, 128])),
    'soft_updates':  _dqn_cfg(learning_rate=3e-4, tau=0.005,
                              policy_kwargs=dict(net_arch=[128, 128])),
    'less_explore':  _dqn_cfg(learning_rate=3e-4, tau=0.005,
                              exploration_fraction=0.5, exploration_final_eps=0.02,
                              policy_kwargs=dict(net_arch=[128, 128])),
}

PPO_CONFIGS: Dict[str, dict] = {
    'baseline':       _ppo_cfg(),
    'lr_lower':       _ppo_cfg(learning_rate=1e-4),
    'bigger_net':     _ppo_cfg(policy_kwargs=dict(net_arch=[128, 128])),
    'more_entropy':   _ppo_cfg(ent_coef=0.05,
                               policy_kwargs=dict(net_arch=[128, 128])),
    'longer_rollout': _ppo_cfg(ent_coef=0.05, n_steps=4096,
                               policy_kwargs=dict(net_arch=[128, 128])),
    'tighter_clip':   _ppo_cfg(ent_coef=0.05, n_steps=4096, clip_range=0.1,
                               policy_kwargs=dict(net_arch=[128, 128])),
}



# 2. Model construction
def _build_model(algo: str, env, cfg: dict, seed: int, verbose: int):
    if algo == 'dqn':
        from stable_baselines3 import DQN
        return DQN('MlpPolicy', env=env, seed=seed, verbose=verbose,
                   device='cpu', **cfg)
    if algo == 'ppo':
        from stable_baselines3 import PPO
        return PPO('MlpPolicy', env=env, seed=seed, verbose=verbose,
                   device='cpu', **cfg)
    raise ValueError(f"Unknown algo: {algo!r}. Use 'dqn' or 'ppo'.")



# 3. Stage 1 — Sweep
def run_sweep(
    algo: str,
    configs: Dict[str, dict],
    total_timesteps: int = 100_000,
    eval_episodes: int = 500,
    seed: int = 42,
    save_csv: bool = True,
) -> pd.DataFrame:
    """
    Train + evaluate every configuration in `configs`.

    Returns a DataFrame ranked by overall_survival (descending).
    Side-effect: writes sweep_<algo>_results.csv to the project root.

    Per-config: ~2 min @ 100k steps on CPU. 6 configs ≈ 12 min per algo.
    """
    rows = []
    for name, cfg in configs.items():
        log_dir = os.path.join(LOG_ROOT, f'sweep_{algo}_{name}')
        env = make_train_env(seed=seed, log_dir=log_dir)
        model = _build_model(algo, env, cfg, seed=seed, verbose=0)

        t0 = time.time()
        model.learn(total_timesteps=total_timesteps, progress_bar=False)
        train_time = time.time() - t0
        env.close()

        eval_result = evaluate_agent(model, n_episodes=eval_episodes,
                                     seed=123, deterministic=True)

        rows.append({
            'config':           name,
            'survival':         eval_result['overall_survival'],
            'mean_return':      eval_result['mean_return'],
            'noisy_survival':   eval_result['noisy_survival'],
            'clean_survival':   eval_result['clean_survival'],
            'missing_survival': eval_result['missing_survival'],
            'train_time_s':     round(train_time, 1),
        })

        print(f'[{algo:3s}] {name:14s}  '
              f'survival={eval_result["overall_survival"]*100:5.1f}%  '
              f'return={eval_result["mean_return"]:.3f}  '
              f'time={train_time:5.1f}s')

    df = pd.DataFrame(rows).sort_values('survival', ascending=False).reset_index(drop=True)

    if save_csv:
        df.to_csv(f'sweep_{algo}_results.csv', index=False)

    return df



# 4. Stage 2 — Train best config at higher budget
def train_best(
    algo: str,
    config_name: str,
    total_timesteps: int = 500_000,
    seed: int = 42,
    verbose: int = 1,
    save: bool = True,
):
    """
    Train the chosen configuration at the full budget for the report's
    headline results.

    Returns:
        (model, training_log_df) — same shape as configb.train_dqn().
    """
    configs = DQN_CONFIGS if algo == 'dqn' else PPO_CONFIGS
    if config_name not in configs:
        raise KeyError(f"Unknown config {config_name!r} for {algo}. "
                       f"Available: {list(configs)}")
    cfg = configs[config_name]

    log_dir = os.path.join(LOG_ROOT, f'{algo}_final_{config_name}_seed{seed}')
    env = make_train_env(seed=seed, log_dir=log_dir)
    model = _build_model(algo, env, cfg, seed=seed, verbose=verbose)

    print(f'Training {algo.upper()} ({config_name}) for {total_timesteps:,} steps...')
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    print(f'Done in {time.time() - t0:.1f}s')
    env.close()

    monitor_csv = os.path.join(log_dir, 'monitor.monitor.csv')
    training_log = pd.read_csv(monitor_csv, skiprows=1)

    if save:
        os.makedirs(MODEL_ROOT, exist_ok=True)
        model.save(os.path.join(MODEL_ROOT, f'{algo}_final_{config_name}'))

    return model, training_log
