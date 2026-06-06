"""
configb.py — Config B (continuous observation) helpers for the RL Sepsis project.

This module owns all the heavy lifting for Configuration B:
  - Environment construction (training + evaluation)
  - Training loops for DQN and PPO
  - Agent evaluation with clinical-subgroup breakdown
  - Plotting and results-table helpers

The notebook should only IMPORT from here and call these functions.
Keeping code out of the notebook makes the notebook readable and the
project easy to debug, review, and reproduce.
"""

from __future__ import annotations

import os
from typing import Optional

import numpy as np
import pandas as pd

from envs.wrappers import make_clinical_env

# Folders where Monitor CSVs and trained models are written
LOG_ROOT = 'logs_configB'
MODEL_ROOT = 'models_configB'


# 1. Environment construction

def make_train_env(seed: int = 42, log_dir: Optional[str] = None):
    """
    Build the Config B training environment.

    Wraps `make_clinical_env()` (47-dim continuous obs + clinical reality
    wrappers: noise, missing values, acute events) inside SB3's `Monitor`,
    which logs (episode_return, episode_length) for every finished episode.

    Args:
        seed     : RNG seed for the action space (reproducibility).
        log_dir  : Folder where Monitor writes its CSV. If None, in-memory only.

    Returns:
        gymnasium.Env wrapped in stable_baselines3.common.monitor.Monitor.
    """
    from stable_baselines3.common.monitor import Monitor

    env = make_clinical_env()
    env.action_space.seed(seed)

    if log_dir is not None:
        os.makedirs(log_dir, exist_ok=True)
        env = Monitor(env, filename=os.path.join(log_dir, 'monitor'))
    else:
        env = Monitor(env)

    return env


# 2. Training — DQN

def train_dqn(
    total_timesteps: int = 100_000,
    seed: int = 42,
    verbose: int = 1,
    save: bool = True,
):
    """
    Train a DQN agent on the Config B clinical environment.

    DQN approximates Q(s, a) with a neural net and learns off-policy from a
    replay buffer with ε-greedy exploration and a target network for
    stability. See `references` section in the report for the full citation.

    Args:
        total_timesteps : Number of environment steps (not episodes).
        seed            : Master seed. Set for env, numpy, torch.
        verbose         : 0=silent, 1=info, 2=debug (passed to SB3).
        save            : If True, save model + training log to MODEL_ROOT/LOG_ROOT.

    Returns:
        (model, training_log_df)
          model           : trained stable_baselines3.DQN instance
          training_log_df : pandas DataFrame with columns [r, l, t]
                            r = episode return, l = length, t = time elapsed
    """
    from stable_baselines3 import DQN

    log_dir = os.path.join(LOG_ROOT, f'dqn_seed{seed}')
    env = make_train_env(seed=seed, log_dir=log_dir)

    model = DQN(
        policy='MlpPolicy',
        env=env,
        learning_rate=1e-4,
        buffer_size=50_000,
        learning_starts=1_000,
        batch_size=64,
        tau=1.0,                       # hard target update (standard DQN)
        gamma=1.0,                     # imposed by the ICU-Sepsis paper
        train_freq=4,
        gradient_steps=1,
        target_update_interval=500,
        exploration_fraction=0.3,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=verbose,
        device='cpu',                  # small net, CPU is faster than GPU here
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    env.close()

    # Read back the Monitor CSV as a DataFrame
    monitor_csv = os.path.join(log_dir, 'monitor.monitor.csv')
    training_log = pd.read_csv(monitor_csv, skiprows=1)

    if save:
        os.makedirs(MODEL_ROOT, exist_ok=True)
        model.save(os.path.join(MODEL_ROOT, f'dqn_seed{seed}'))

    return model, training_log


# 2b. Training — PPO

def train_ppo(
    total_timesteps: int = 100_000,
    seed: int = 42,
    verbose: int = 1,
    save: bool = True,
):
    """
    Train a PPO agent on the Config B clinical environment.

    PPO is an on-policy actor-critic method. Two networks share an MLP trunk:
      - actor : outputs a categorical distribution over the 25 actions
      - critic: outputs V(s)
    Updates are clipped (the 'proximal' part) so each policy update stays
    close to the previous one — much more stable than vanilla policy gradient.

    Args:
        total_timesteps : Number of environment steps.
        seed            : Master seed.
        verbose         : 0=silent, 1=info.
        save            : Whether to save model and load log from disk.

    Returns:
        (model, training_log_df) — same shape as train_dqn().
    """
    from stable_baselines3 import PPO

    log_dir = os.path.join(LOG_ROOT, f'ppo_seed{seed}')
    env = make_train_env(seed=seed, log_dir=log_dir)

    model = PPO(
        policy='MlpPolicy',
        env=env,
        learning_rate=3e-4,
        n_steps=2048,                  # rollout length before each update
        batch_size=64,
        n_epochs=10,                   # optimisation passes per rollout
        gamma=1.0,                     # imposed by the ICU-Sepsis paper
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.01,                 # entropy bonus — helps sparse rewards
        vf_coef=0.5,
        max_grad_norm=0.5,
        policy_kwargs=dict(net_arch=[64, 64]),
        seed=seed,
        verbose=verbose,
        device='cpu',
    )

    model.learn(total_timesteps=total_timesteps, progress_bar=False)
    env.close()

    monitor_csv = os.path.join(log_dir, 'monitor.monitor.csv')
    training_log = pd.read_csv(monitor_csv, skiprows=1)

    if save:
        os.makedirs(MODEL_ROOT, exist_ok=True)
        model.save(os.path.join(MODEL_ROOT, f'ppo_seed{seed}'))

    return model, training_log


# =============================================================================
# 3. Evaluation

def evaluate_agent(
    model,
    n_episodes: int = 1000,
    seed: int = 123,
    deterministic: bool = True,
) -> dict:
    """
    Run a trained agent (or random policy) for `n_episodes` and compute
    survival rate broken down by clinical subgroup.

    Subgroups come from `info` keys added by the clinical wrappers:
      - 'noisy_episode'    : True if this episode had noisy observations
      - 'missing_features' : list of missing feature indices (or None)
      - 'acute_event'      : True if an acute event killed the patient

    Args:
        model         : SB3 model with `.predict(obs)`, or None for random policy.
        n_episodes    : Number of evaluation episodes.
        seed          : Master seed (each episode uses a derived seed).
        deterministic : If True, model picks argmax action (no exploration).

    Returns:
        dict with the metrics required to build the results table and plots.
    """
    env = make_clinical_env()
    rng = np.random.default_rng(seed)

    returns, lengths = [], []
    noisy_returns, clean_returns = [], []
    missing_returns, no_missing_returns = [], []
    acute_count = 0

    for ep in range(n_episodes):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        ep_return, ep_length, done = 0.0, 0, False
        ep_noisy = info.get('noisy_episode', False)
        ep_missing = info.get('missing_features') is not None
        ep_acute = False

        while not done:
            if model is None:
                action = env.action_space.sample()
            else:
                action, _ = model.predict(obs, deterministic=deterministic)
                action = int(action)

            obs, reward, terminated, truncated, info = env.step(action)
            ep_return += reward
            ep_length += 1
            done = terminated or truncated
            if info.get('acute_event', False):
                ep_acute = True

        returns.append(ep_return)
        lengths.append(ep_length)
        (noisy_returns if ep_noisy else clean_returns).append(ep_return)
        (missing_returns if ep_missing else no_missing_returns).append(ep_return)
        if ep_acute:
            acute_count += 1

    env.close()

    def survival_rate(rs):
        return float(np.mean(np.array(rs) > 0)) if len(rs) else float('nan')

    return {
        'overall_survival':     survival_rate(returns),
        'noisy_survival':       survival_rate(noisy_returns),
        'clean_survival':       survival_rate(clean_returns),
        'missing_survival':     survival_rate(missing_returns),
        'no_missing_survival':  survival_rate(no_missing_returns),
        'acute_episode_rate':   acute_count / n_episodes,
        'mean_return':          float(np.mean(returns)),
        'mean_length':          float(np.mean(lengths)),
        'returns':              np.array(returns),
        'n_noisy':              len(noisy_returns),
        'n_missing':            len(missing_returns),
    }


# =============================================================================
# 4. Plotting & results table
# =============================================================================

# Reference baselines measured elsewhere in the project (for plot guide lines)
RANDOM_SURVIVAL = 0.659   # random policy on the clinical env
EXPERT_SURVIVAL = 0.679   # MIMIC-III clinician policy on the clinical env


def _binned_survival(df, n_bins):
    """Helper: survival rate per timestep bucket. Returns (centers, rates)."""
    steps = df['l'].cumsum().values
    survived = (df['r'] > 0).astype(float).values
    total = steps[-1]
    edges = np.linspace(0, total, n_bins + 1)
    idx = np.clip(np.digitize(steps, edges) - 1, 0, n_bins - 1)
    centers, rates = [], []
    for b in range(n_bins):
        mask = idx == b
        if mask.sum() >= 5:
            centers.append((edges[b] + edges[b + 1]) / 2)
            rates.append(survived[mask].mean() * 100)
    return np.array(centers), np.array(rates)


def plot_learning_curves_grid(logs: dict, n_bins: int = 40, figsize=(15, 4.5),
                              save_path: str = None, ylim=(55, 75)):
    """
    One subplot per model — each model's learning curve in isolation, so the
    rise (or plateau) of every algorithm is clearly visible without overlap.

    Adds a linear trend line per model to make the learning direction obvious,
    plus random/expert reference lines. Same y-axis across panels for fair
    visual comparison.

    Args:
        logs : dict {model_name: training_log_df} with columns r, l.
        n_bins : timestep buckets for the survival curve.
    """
    import matplotlib.pyplot as plt

    names = list(logs)
    fig, axes = plt.subplots(1, len(names), figsize=figsize, sharey=True)
    if len(names) == 1:
        axes = [axes]
    colors = plt.cm.tab10.colors

    for i, (name, df) in enumerate(logs.items()):
        ax = axes[i]
        centers, rates = _binned_survival(df, n_bins)
        ax.plot(centers, rates, color=colors[i % 10], linewidth=1.6,
                marker='o', markersize=3, alpha=0.7, label='Survival')
        # linear trend line — shows learning direction at a glance
        if len(centers) > 2:
            z = np.polyfit(centers, rates, 1)
            ax.plot(centers, np.poly1d(z)(centers), color='black',
                    linewidth=2.2, ls='-', label=f'Trend ({z[0]*1e6:+.1f}%/M)')

        ax.axhline(RANDOM_SURVIVAL * 100, ls='--', color='grey', alpha=0.8)
        ax.axhline(EXPERT_SURVIVAL * 100, ls=':', color='red', alpha=0.7)
        ax.set_title(name, fontweight='bold')
        ax.set_xlabel('Training timesteps')
        if i == 0:
            ax.set_ylabel('Survival rate (%)')
        if ylim:
            ax.set_ylim(*ylim)
        ax.legend(fontsize=8, loc='lower right')
        ax.grid(True, alpha=0.3)

    fig.suptitle('Config B — Learning curves per model '
                 '(grey=random, red=clinician expert)', fontweight='bold')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
    return fig


def plot_learning_curves(logs: dict, n_bins: int = 40, figsize=(11, 6),
                         save_path: str = None, ylim=(55, 75)):
    """
    Plot survival rate vs training timesteps for each trained model.

    Uses TIMESTEP BINNING (not per-episode rolling) for a clean, readable
    curve: training is split into `n_bins` equal-width timestep buckets and
    the survival rate is computed within each bucket. This removes the
    high-frequency noise that makes per-episode survival (a binary signal
    over ~9-step episodes) look like spaghetti.

    A rising curve = the agent is learning. A flat curve = no learning.

    Args:
        logs : dict {model_name: training_log_df} with columns r (return),
               l (episode length).
        n_bins : number of timestep buckets (40 → smooth but responsive).
        ylim : y-axis range; default zooms to the region where the action is.
        save_path : if given, save the figure there.
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors

    for i, (name, df) in enumerate(logs.items()):
        steps = df['l'].cumsum().values
        survived = (df['r'] > 0).astype(float).values
        total = steps[-1]
        edges = np.linspace(0, total, n_bins + 1)
        # bucket index for each episode by its cumulative step
        idx = np.clip(np.digitize(steps, edges) - 1, 0, n_bins - 1)
        centers, rates = [], []
        for b in range(n_bins):
            mask = idx == b
            if mask.sum() >= 5:                      # need enough episodes
                centers.append((edges[b] + edges[b + 1]) / 2)
                rates.append(survived[mask].mean() * 100)
        ax.plot(centers, rates, label=name, linewidth=2.2,
                color=colors[i % 10], marker='o', markersize=3)

    ax.axhline(RANDOM_SURVIVAL * 100, ls='--', color='grey', alpha=0.9,
               label=f'Random ({RANDOM_SURVIVAL*100:.1f}%)')
    ax.axhline(EXPERT_SURVIVAL * 100, ls=':', color='black', alpha=0.9,
               label=f'Clinician expert ({EXPERT_SURVIVAL*100:.1f}%)')

    ax.set_xlabel('Training timesteps')
    ax.set_ylabel('Survival rate (%)')
    ax.set_title('Config B — Learning curves (survival per training phase)',
                 fontweight='bold')
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend(loc='lower right', fontsize=9, framealpha=0.95)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
    return fig


def plot_subgroup_survival(eval_results: dict, figsize=(11, 6),
                           save_path: str = None):
    """
    Grouped bar chart: survival rate per clinical subgroup, per model.

    Args:
        eval_results : dict {model_name: evaluate_agent(...) output dict}.
    """
    import matplotlib.pyplot as plt

    subgroups = ['overall_survival', 'clean_survival', 'noisy_survival',
                 'no_missing_survival', 'missing_survival']
    labels = ['Overall', 'Clean', 'Noisy', 'No-missing', 'Missing']

    names = list(eval_results)
    n_models = len(names)
    x = np.arange(len(subgroups))
    width = 0.8 / max(n_models, 1)

    fig, ax = plt.subplots(figsize=figsize)
    for i, name in enumerate(names):
        vals = [eval_results[name][k] * 100 for k in subgroups]
        bars = ax.bar(x + i * width, vals, width, label=name)
        ax.bar_label(bars, fmt='%.0f', fontsize=7, padding=2)

    ax.axhline(RANDOM_SURVIVAL * 100, ls='--', color='grey', alpha=0.8,
               label=f'Random ({RANDOM_SURVIVAL*100:.1f}%)')
    ax.set_xticks(x + width * (n_models - 1) / 2)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Survival rate (%)')
    ax.set_ylim(55, 78)   # zoom to where the differences live
    ax.set_title('Config B — Survival by clinical subgroup', fontweight='bold')
    ax.legend(fontsize=9, ncol=2, loc='upper right')
    ax.grid(True, axis='y', alpha=0.3)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
    return fig


def build_results_table(eval_results: dict, include_baselines: bool = True):
    """
    Build a comparison DataFrame from evaluate_agent outputs.

    Args:
        eval_results : dict {model_name: evaluate_agent(...) output}.
        include_baselines : prepend random + clinician-expert reference rows.

    Returns:
        pandas DataFrame (survival rates as percentages).
    """
    rows = []
    if include_baselines:
        rows.append({'Model': 'Random baseline', 'Overall %': RANDOM_SURVIVAL * 100,
                     'Clean %': np.nan, 'Noisy %': np.nan,
                     'No-missing %': np.nan, 'Missing %': np.nan, 'Mean return': np.nan})
        rows.append({'Model': 'Clinician expert', 'Overall %': EXPERT_SURVIVAL * 100,
                     'Clean %': np.nan, 'Noisy %': np.nan,
                     'No-missing %': np.nan, 'Missing %': np.nan, 'Mean return': np.nan})

    for name, r in eval_results.items():
        rows.append({
            'Model':        name,
            'Overall %':    r['overall_survival'] * 100,
            'Clean %':      r['clean_survival'] * 100,
            'Noisy %':      r['noisy_survival'] * 100,
            'No-missing %': r['no_missing_survival'] * 100,
            'Missing %':    r['missing_survival'] * 100,
            'Mean return':  r['mean_return'],
        })

    df = pd.DataFrame(rows)
    return df.round(2)


# =============================================================================
# 5. Convergence metric & exploration/exploitation diagnostics
# =============================================================================

def convergence_step(log_df, threshold: float = EXPERT_SURVIVAL,
                     window: int = 2000) -> int | None:
    """
    First cumulative training timestep at which the rolling-window survival
    rate (`window` episodes) reaches `threshold`.

    Pedagogical metric: 'how many steps until the agent reaches the clinician
    expert level (~67.9%)?' A small number = fast learner; None = never crossed.

    Args:
        log_df    : training log (Monitor CSV) with columns r, l.
        threshold : survival rate to cross (fraction, e.g. 0.679 = expert).
        window    : rolling window in episodes for smoothing the noise.

    Returns:
        int cumulative timestep, or None if the threshold is never crossed.
    """
    steps = log_df['l'].cumsum().values
    survived = (log_df['r'] > 0).astype(float)
    rolling = survived.rolling(window, min_periods=window // 2).mean().values
    above = np.where(rolling >= threshold)[0]
    if len(above) == 0:
        return None
    return int(steps[above[0]])


def plot_exploration_schedules(dqn_params: dict, total_timesteps: int = 1_000_000,
                               figsize=(11, 4.5), save_path: str = None):
    """
    Plot the analytical ε-greedy schedules of DQN-family algorithms.

    SB3 decays ε linearly from `exploration_initial_eps` to
    `exploration_final_eps` over the first `exploration_fraction *
    total_timesteps` steps, then holds at the final value. Visualising the
    schedule answers the rubric's 'exploration vs exploitation' question
    for value-based agents without needing per-step logs.

    Args:
        dqn_params : dict {model_name: best_params dict} for DQN/QR-DQN.
        total_timesteps : the training budget used (1M in our finals).
    """
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10.colors
    for i, (name, p) in enumerate(dqn_params.items()):
        eps0 = p.get('exploration_initial_eps', 1.0)
        eps_final = p.get('exploration_final_eps', 0.05)
        frac = p.get('exploration_fraction', 0.3)
        decay_end = frac * total_timesteps
        steps = np.linspace(0, total_timesteps, 1000)
        eps = np.where(steps < decay_end,
                       eps0 - (eps0 - eps_final) * (steps / max(decay_end, 1)),
                       eps_final)
        ax.plot(steps, eps, color=colors[i % 10], linewidth=2.2,
                label=f'{name} (final ε={eps_final:.3f}, fraction={frac:.2f})')

    ax.set_xlabel('Training timesteps')
    ax.set_ylabel('Exploration rate ε')
    ax.set_title('Config B — ε-greedy schedules (value-based agents)',
                 fontweight='bold')
    ax.legend(loc='upper right', fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', dpi=120)
    plt.show()
    return fig


def policy_entropy(model, n_samples: int = 500, seed: int = 42) -> float | None:
    """
    Mean Shannon entropy (in nats) of the model's policy over a sample of
    states from the clinical env. Captures exploitation-side stochasticity
    for stochastic policies (PPO).

    For DQN/QR-DQN the policy at eval is the argmax → entropy is trivially
    0, so the function returns None for those.

    Max entropy for 25 uniform actions = ln(25) ≈ 3.22.
    """
    import torch

    if not hasattr(model.policy, 'get_distribution'):
        return None     # argmax-based policy (DQN, QR-DQN)

    env = make_clinical_env()
    rng = np.random.default_rng(seed)
    obs_list = []
    for _ in range(60):
        obs, info = env.reset(seed=int(rng.integers(0, 1_000_000)))
        done = False
        while not done and len(obs_list) < n_samples:
            obs_list.append(obs.copy())
            action, _ = model.predict(obs, deterministic=False)
            obs, r, te, tr, info = env.step(int(action))
            done = te or tr
        if len(obs_list) >= n_samples:
            break
    env.close()

    obs_arr = np.array(obs_list[:n_samples], dtype=np.float32)
    obs_t = torch.as_tensor(obs_arr)
    with torch.no_grad():
        dist = model.policy.get_distribution(obs_t)
        ent = dist.entropy().detach().cpu().numpy()
    return float(np.mean(ent))
