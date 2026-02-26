from pogema_toolbox.results_holder import ResultsHolder


def run_episode(env, algo):
    """
    Runs an episode in the environment using the given algorithm.

    Args:
        env: The environment to run the episode in.
        algo: The algorithm used for action selection.

    Returns:
        ResultsHolder: Object containing the results of the episode.
    """
    algo.reset_states()
    results_holder = ResultsHolder()

    obs, _ = env.reset(seed=env.unwrapped.grid_config.seed)
    while True:
        obs, rew, terminated, truncated, infos = env.step(algo.act(obs))
        results_holder.after_step(infos)

        if all(terminated) or all(truncated):
            break
    return results_holder.get_final()


def run_batch_episodes(envs, algo, per_env_algos=None):
    """
    Run multiple envs in lockstep, batching algo calls via act_batch when available.

    Args:
        envs: List of environments to run simultaneously.
        algo: The algorithm used for action selection. If it has an act_batch method,
              observations are batched into a single call for all active envs.
        per_env_algos: Optional list of per-env algorithm instances (same length as envs).
                       Used as fallback when algo lacks act_batch — each env gets its own
                       algo instance to avoid stateful conflicts.

    Returns:
        list: Per-env final results from ResultsHolder.
    """
    use_batch = hasattr(algo, 'act_batch')
    results_holders = [ResultsHolder() for _ in envs]

    all_obs = []
    for env in envs:
        obs, _ = env.reset(seed=env.unwrapped.grid_config.seed)
        all_obs.append(obs)

    active = [True] * len(envs)

    while any(active):
        active_obs = []
        active_positions = []
        for i, (obs, is_active) in enumerate(zip(all_obs, active)):
            if is_active:
                active_obs.append(obs)
                active_positions.append(i)

        if use_batch:
            active_actions = algo.act_batch(active_obs, active_positions)
        else:
            active_actions = [per_env_algos[pos].act(obs) for pos, obs in zip(active_positions, active_obs)]

        for pos, actions in zip(active_positions, active_actions):
            obs, rew, terminated, truncated, infos = envs[pos].step(actions)
            all_obs[pos] = obs
            results_holders[pos].after_step(infos)
            if all(terminated) or all(truncated):
                active[pos] = False

    return [rh.get_final() for rh in results_holders]
