"""
configb_optuna.py — Bayesian hyperparameter optimization for Config B.

Replaces the manual sweep with Optuna (TPE sampler) + ASHA pruning
(SuccessiveHalvingPruner). Supports three algorithms:
  - 'dqn'   : stable_baselines3.DQN          (value-based, off-policy)
  - 'qrdqn' : sb3_contrib.QRDQN              (distributional DQN)
  - 'ppo'   : stable_baselines3.PPO          (policy-based, on-policy)

How it works
------------
Each trial samples a hyperparameter config and trains for up to
`search_budget` steps on the clinical env. Every `eval_every` steps the
agent is evaluated deterministically on `eval_episodes` episodes; that
survival rate is reported to Optuna. The ASHA pruner kills trials whose
intermediate survival is in the bottom fraction, so compute concentrates
on promising configs.

After the study, retrain the best config at full budget (e.g. 1M steps)
with `train_final`.

Usage from the notebook
-----------------------
    from envs.configb_optuna import optimize, train_final

    study_dqn = optimize('dqn', n_trials=20, search_budget=300_000)
    print(study_dqn.best_value, study_dqn.best_params)

    model_dqn, log_dqn = train_final('dqn', study_dqn.best_params,
                                     total_timesteps=1_000_000)
"""

from __future__ import annotations

import os
import json
import time
from typing import Optional

import numpy as np
import pandas as pd
import optuna
from optuna.pruners import SuccessiveHalvingPruner
from optuna.samplers import TPESampler

from stable_baselines3.common.callbacks import BaseCallback

from envs.configb import make_train_env, evaluate_agent, LOG_ROOT, MODEL_ROOT

GAMMA = 1.0  # imposed by the ICU-Sepsis paper — never tuned

# Persistent storage: studies survive kernel restarts, best params saved to JSON
OPTUNA_DB = 'sqlite:///optuna_configB.db'
BEST_PARAMS_DIR = 'best_params_configB'


def load_best_params(algo: str) -> dict:
    """Load the best params saved by a previous `optimize(algo, ...)` run."""
    path = os.path.join(BEST_PARAMS_DIR, f'{algo}_best_params.json')
    with open(path, encoding='utf-8') as f:
        return json.load(f)

_NET_ARCHS = {
    'small':  [64, 64],
    'medium': [128, 128],
    'large':  [256, 256],
    'deep':   [256, 256, 256],
}


# =============================================================================
# 1. Model construction from a sampled config
# =============================================================================

def _build(algo: str, env, params: dict, seed: int, verbose: int = 0):
    """Instantiate a model from a flat params dict (as sampled by Optuna)."""
    p = dict(params)  # copy
    net_arch = _NET_ARCHS[p.pop('net_arch_key', 'medium')]

    if algo in ('dqn', 'qrdqn'):
        policy_kwargs = dict(net_arch=net_arch)
        if algo == 'qrdqn':
            policy_kwargs['n_quantiles'] = p.pop('n_quantiles', 50)
        common = dict(
            policy='MlpPolicy', env=env, gamma=GAMMA, seed=seed,
            verbose=verbose, device='cpu', policy_kwargs=policy_kwargs, **p,
        )
        if algo == 'dqn':
            from stable_baselines3 import DQN
            return DQN(**common)
        from sb3_contrib import QRDQN
        return QRDQN(**common)

    if algo == 'ppo':
        from stable_baselines3 import PPO
        return PPO(
            policy='MlpPolicy', env=env, gamma=GAMMA, seed=seed,
            verbose=verbose, device='cpu',
            policy_kwargs=dict(net_arch=net_arch), **p,
        )

    raise ValueError(f"Unknown algo {algo!r}")


# =============================================================================
# 2. Search spaces
# =============================================================================

def _sample_params(algo: str, trial: optuna.Trial) -> dict:
    if algo in ('dqn', 'qrdqn'):
        params = dict(
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 5e-3, log=True),
            buffer_size=trial.suggest_categorical('buffer_size', [50_000, 100_000, 200_000]),
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            learning_starts=trial.suggest_categorical('learning_starts', [1_000, 5_000]),
            tau=trial.suggest_categorical('tau', [1.0, 0.01, 0.005]),
            train_freq=trial.suggest_categorical('train_freq', [1, 4, 8, 16]),
            gradient_steps=trial.suggest_categorical('gradient_steps', [1, 2, 4]),
            target_update_interval=trial.suggest_categorical('target_update_interval', [250, 500, 1000, 2000]),
            exploration_fraction=trial.suggest_float('exploration_fraction', 0.1, 0.6),
            exploration_final_eps=trial.suggest_float('exploration_final_eps', 0.01, 0.10),
            net_arch_key=trial.suggest_categorical('net_arch_key', ['small', 'medium', 'large']),
        )
        if algo == 'qrdqn':
            params['n_quantiles'] = trial.suggest_categorical('n_quantiles', [25, 50, 100, 200])
        return params

    if algo == 'ppo':
        return dict(
            learning_rate=trial.suggest_float('learning_rate', 1e-5, 1e-3, log=True),
            n_steps=trial.suggest_categorical('n_steps', [1024, 2048, 4096]),
            batch_size=trial.suggest_categorical('batch_size', [32, 64, 128, 256]),
            n_epochs=trial.suggest_categorical('n_epochs', [3, 5, 10, 20]),
            gae_lambda=trial.suggest_float('gae_lambda', 0.90, 0.99),
            clip_range=trial.suggest_categorical('clip_range', [0.1, 0.2, 0.3]),
            ent_coef=trial.suggest_float('ent_coef', 1e-4, 0.1, log=True),
            vf_coef=trial.suggest_float('vf_coef', 0.3, 0.7),
            max_grad_norm=trial.suggest_categorical('max_grad_norm', [0.3, 0.5, 1.0]),
            net_arch_key=trial.suggest_categorical('net_arch_key', ['small', 'medium', 'large']),
        )

    raise ValueError(f"Unknown algo {algo!r}")


# =============================================================================
# 3. Pruning callback — report intermediate survival to Optuna
# =============================================================================

class _PruningCallback(BaseCallback):
    """
    Every `eval_every` steps: evaluate and report to the Optuna trial (for
    ASHA pruning). Also enforces a per-trial wall-clock timeout so that no
    single computationally-expensive config can hang the whole study.
    """

    def __init__(self, trial, eval_every, eval_episodes, max_seconds=None, verbose=0):
        super().__init__(verbose)
        self.trial = trial
        self.eval_every = eval_every
        self.eval_episodes = eval_episodes
        self.max_seconds = max_seconds
        self._next_eval = eval_every
        self._step_idx = 0
        self.last_survival = 0.0
        self._t_start = time.time()

    def _on_step(self) -> bool:
        # Wall-clock timeout: prune slow trials regardless of performance
        if self.max_seconds is not None and (time.time() - self._t_start) > self.max_seconds:
            print(f'  [trial {self.trial.number}] timeout '
                  f'({self.max_seconds}s, {self.num_timesteps} steps) → pruned')
            raise optuna.TrialPruned()

        if self.num_timesteps >= self._next_eval:
            self._next_eval += self.eval_every
            res = evaluate_agent(self.model, n_episodes=self.eval_episodes,
                                 seed=999, deterministic=True)
            self.last_survival = res['overall_survival']
            self.trial.report(self.last_survival, self._step_idx)
            self._step_idx += 1
            if self.trial.should_prune():
                raise optuna.TrialPruned()
        return True


# =============================================================================
# 4. Optimize
# =============================================================================

def optimize(
    algo: str,
    n_trials: int = 20,
    search_budget: int = 300_000,
    eval_every: int = 50_000,
    eval_episodes: int = 300,
    seed: int = 42,
    reduction_factor: int = 3,
    max_trial_seconds: Optional[float] = 900,
    study_name: Optional[str] = None,
) -> optuna.Study:
    """
    Run Bayesian optimization for one algorithm.

    Returns the completed Optuna Study. Best config is in `study.best_params`,
    best survival in `study.best_value`.

    Persistence:
      - Full study saved to SQLite (optuna_configB.db) — survives kernel restarts.
        Re-running with the same study_name resumes (load_if_exists=True).
      - Best params saved to best_params_configB/<algo>_best_params.json.
        Reload later with load_best_params(algo) — no need to re-tune.
      - All trials saved to optuna_<algo>_trials.csv.
    """
    sampler = TPESampler(seed=seed)
    pruner = SuccessiveHalvingPruner(min_resource=1, reduction_factor=reduction_factor)
    study = optuna.create_study(
        direction='maximize', sampler=sampler, pruner=pruner,
        study_name=study_name or f'configB_{algo}',
        storage=OPTUNA_DB, load_if_exists=True,
    )

    def objective(trial: optuna.Trial) -> float:
        params = _sample_params(algo, trial)
        log_dir = os.path.join(LOG_ROOT, f'optuna_{algo}_trial{trial.number}')
        env = make_train_env(seed=seed, log_dir=log_dir)
        model = _build(algo, env, params, seed=seed, verbose=0)
        cb = _PruningCallback(trial, eval_every, eval_episodes,
                              max_seconds=max_trial_seconds)
        try:
            model.learn(total_timesteps=search_budget, callback=cb, progress_bar=False)
        finally:
            env.close()
        # Final full eval for the trial's reported value
        res = evaluate_agent(model, n_episodes=500, seed=123, deterministic=True)
        return res['overall_survival']

    t0 = time.time()
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
    elapsed = time.time() - t0

    df = study.trials_dataframe()
    df.to_csv(f'optuna_{algo}_trials.csv', index=False)

    n_pruned = sum(t.state == optuna.trial.TrialState.PRUNED for t in study.trials)
    n_complete = sum(t.state == optuna.trial.TrialState.COMPLETE for t in study.trials)

    if n_complete == 0:
        print(f'\n[{algo}] WARNING: 0 trials completed ({n_pruned} pruned). '
              f'Try raising max_trial_seconds or search_budget. Nothing saved.')
        return study

    # Persist best params to JSON for easy reload (no re-tuning needed)
    os.makedirs(BEST_PARAMS_DIR, exist_ok=True)
    best_path = os.path.join(BEST_PARAMS_DIR, f'{algo}_best_params.json')
    with open(best_path, 'w', encoding='utf-8') as f:
        json.dump({'best_value': study.best_value,
                   'best_params': study.best_params}, f, indent=2)

    print(f'\n[{algo}] {n_trials} trials in {elapsed/60:.1f} min '
          f'({n_complete} complete, {n_pruned} pruned)')
    print(f'[{algo}] BEST survival = {study.best_value*100:.1f}%')
    print(f'[{algo}] BEST params   = {study.best_params}')
    print(f'[{algo}] Saved to {best_path}')
    return study


# =============================================================================
# 5. Final training of the best config
# =============================================================================

def train_final(
    algo: str,
    best_params: dict,
    total_timesteps: int = 1_000_000,
    seed: int = 42,
    verbose: int = 1,
    save: bool = True,
):
    """
    Retrain the best Optuna config at full budget for the report results.

    Returns (model, training_log_df).
    """
    log_dir = os.path.join(LOG_ROOT, f'{algo}_optuna_final_seed{seed}')
    monitor_csv = os.path.join(log_dir, 'monitor.monitor.csv')
    model_path = os.path.join(MODEL_ROOT, f'{algo}_optuna_final')
    
    # Try loading existing model and logs to save 1h+ of compute!
    if os.path.exists(model_path + '.zip') and os.path.exists(monitor_csv):
        print(f'Found pre-trained {algo.upper()} model and logs. Loading from disk instead of re-training...')
        env = make_train_env(seed=seed, log_dir=None)  # Pass None so it doesn't wipe the existing CSV!
        if algo == 'dqn':
            from stable_baselines3 import DQN
            model = DQN.load(model_path, env=env)
        elif algo == 'qrdqn':
            from sb3_contrib import QRDQN
            model = QRDQN.load(model_path, env=env)
        elif algo == 'ppo':
            from stable_baselines3 import PPO
            model = PPO.load(model_path, env=env)
        training_log = pd.read_csv(monitor_csv, skiprows=1)
        return model, training_log

    env = make_train_env(seed=seed, log_dir=log_dir)
    model = _build(algo, env, best_params, seed=seed, verbose=verbose)

    print(f'Training {algo.upper()} (best Optuna config) for {total_timesteps:,} steps...')
    t0 = time.time()
    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    print(f'Done in {(time.time()-t0)/60:.1f} min')
    env.close()

    training_log = pd.read_csv(monitor_csv, skiprows=1)

    if save:
        os.makedirs(MODEL_ROOT, exist_ok=True)
        model.save(model_path)

    return model, training_log
