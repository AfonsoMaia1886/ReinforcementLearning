import numpy as np
from envs.env_setup import make_sepsis_env, N_STATES, N_ACTIONS

SEED = 42

def evaluate_policy(policy, n_episodes=500, seed=SEED):
    np.random.seed(seed)
    env_e = make_sepsis_env()
    rets = []
    for _ in range(n_episodes):
        obs, _ = env_e.reset(seed=np.random.randint(100_000))
        total_r, done = 0.0, False
        while not done:
            obs, r, te, tr, _ = env_e.step(int(policy[int(obs)]))
            total_r += r; done = te or tr
        rets.append(total_r)
    env_e.close()
    return np.array(rets)

def run_model_free(algo, env_factory, n_episodes=5000, alpha=0.1, gamma=1.0,
                   eps_start=1.0, eps_min=0.01, eps_decay=0.995,
                   Q_init=0.0,
                   alpha_decay=1.0,       # 1.0 = no decay; <1.0 = anneal alpha
                   alpha_min=0.001,       # floor for alpha when using alpha_decay
                   lam=0.0,              # eligibility trace lambda (0 = standard TD)
                   eval_every=500, eval_eps=300, seed=SEED):
    """
    Unified enhanced model-free TD learning.

    algo            : 'qlearning' | 'sarsa' | 'double_qlearning'
                      | 'expected_sarsa' | 'sarsa_lambda' | 'qlearning_lambda'
    alpha           : initial learning rate
    alpha_decay     : multiplicative decay applied to alpha each episode
    alpha_min       : minimum alpha (floor)
    lam             : eligibility trace parameter λ ∈ [0, 1]
                      0  → standard 1-step TD  (no traces)
                      1  → Monte Carlo limit
    Q_init          : optimistic initialisation value
    """
    np.random.seed(seed)
    env = env_factory()

    # ── Q-table(s) ───────────────────────────────────────────────────────────
    Q = np.full((N_STATES, N_ACTIONS), Q_init, dtype=np.float64)
    if algo == 'double_qlearning':
        Q2 = np.full((N_STATES, N_ACTIONS), Q_init, dtype=np.float64)

    eps   = eps_start
    cur_alpha = alpha

    ep_returns, eps_log = [], []
    eval_points, eval_sr, eval_mr = [], [] , []
    q_convergence = []

    for ep in range(n_episodes):
        obs, _ = env.reset()
        s = int(obs)
        done = False; total_r = 0.0
        Q_old = Q.copy()

        # Eligibility traces reset at start of each episode
        if algo in ('sarsa_lambda', 'qlearning_lambda'):
            E = np.zeros((N_STATES, N_ACTIONS), dtype=np.float64)

        # Choose first action
        if np.random.rand() < eps:
            a = env.action_space.sample()
        else:
            a = int(np.argmax(Q[s]))

        while not done:
            next_obs, r, te, tr, _ = env.step(a)
            s2 = int(next_obs)
            done = te or tr

            # ── Greedy next action (used by most variants) ─────────────────
            a2_greedy = int(np.argmax(Q[s2]))

            # ── Epsilon-greedy next action (for on-policy methods) ─────────
            if np.random.rand() < eps:
                a2 = env.action_space.sample()
            else:
                a2 = a2_greedy

            # ── TD Target per algorithm ────────────────────────────────────
            if algo == 'sarsa':
                td_target = r + gamma * Q[s2, a2]
                td_error  = td_target - Q[s, a]
                Q[s, a] += cur_alpha * td_error

            elif algo == 'qlearning':
                td_target = r + gamma * float(np.max(Q[s2]))
                td_error  = td_target - Q[s, a]
                Q[s, a] += cur_alpha * td_error

            elif algo == 'double_qlearning':
                # Randomly pick which table selects action vs evaluates it
                if np.random.rand() < 0.5:
                    a_max = int(np.argmax(Q[s2]))   # Q1 selects action
                    td_target = r + gamma * Q2[s2, a_max]  # Q2 evaluates it
                    td_error  = td_target - Q[s, a]
                    Q[s, a] += cur_alpha * td_error
                else:
                    a_max = int(np.argmax(Q2[s2]))  # Q2 selects action
                    td_target = r + gamma * Q[s2, a_max]   # Q1 evaluates it
                    td_error  = td_target - Q2[s, a]
                    Q2[s, a] += cur_alpha * td_error

            elif algo == 'expected_sarsa':
                # Expected value under current eps-greedy policy
                q_next = Q[s2]
                greedy_val = float(np.max(q_next))
                mean_val   = float(np.mean(q_next))
                expected_q = (1 - eps) * greedy_val + eps * mean_val
                td_target  = r + gamma * expected_q
                td_error   = td_target - Q[s, a]
                Q[s, a] += cur_alpha * td_error

            elif algo == 'sarsa_lambda':
                td_error = r + gamma * Q[s2, a2] - Q[s, a]
                E[s, a] += 1.0          # accumulating traces
                Q  += cur_alpha * td_error * E
                # Decay traces: if not greedy, reset (replacing trace variant)
                E  *= gamma * lam

            elif algo == 'qlearning_lambda':  # Watkins Q(λ)
                is_greedy = (a2 == a2_greedy)
                td_target = r + gamma * float(np.max(Q[s2]))
                td_error  = td_target - Q[s, a]
                E[s, a] += 1.0
                Q  += cur_alpha * td_error * E
                if is_greedy:
                    E *= gamma * lam
                else:
                    E[:] = 0.0   # Watkins: cut traces on non-greedy action

            s = s2; a = a2
            total_r += r

        ep_returns.append(total_r)
        eps_log.append(eps)
        eps = max(eps_min, eps * eps_decay)
        cur_alpha = max(alpha_min, cur_alpha * alpha_decay)

        # Q-value convergence diagnostic
        q_convergence.append(float(np.mean(np.abs(Q - Q_old))))

        if (ep + 1) % eval_every == 0:
            if algo == 'double_qlearning':
                greedy = np.argmax((Q + Q2) / 2, axis=1)
            else:
                greedy = np.argmax(Q, axis=1)
            rets = evaluate_policy(greedy, n_episodes=eval_eps)
            eval_sr.append(float(np.mean(rets > 0)) * 100)
            eval_mr.append(float(np.mean(rets)))
            eval_points.append(ep + 1)

    env.close()

    if algo == 'double_qlearning':
        policy = np.argmax((Q + Q2) / 2, axis=1)
    else:
        policy = np.argmax(Q, axis=1)

    metrics = {
        'episode_returns': np.array(ep_returns),
        'epsilons':        np.array(eps_log),
        'eval_points':     np.array(eval_points),
        'eval_sr':         np.array(eval_sr),
        'eval_mr':         np.array(eval_mr),
        'q_convergence':   np.array(q_convergence),
    }
    return policy, Q, metrics

ALGOS_TRYHARD = ['double_qlearning', 'expected_sarsa', 'sarsa_lambda', 'qlearning_lambda']
