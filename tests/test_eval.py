import json
import random

from pogema import BatchAStarAgent, AStarAgent

from pogema_toolbox.create_env import create_env_base, Environment
from pogema_toolbox.evaluator import evaluation
from pogema_toolbox.registry import ToolboxRegistry


class RandomAgent:
    """Random agent with act_batch for batched evaluation testing."""

    def __init__(self, seed=42):
        self.rng = random.Random(seed)

    def act(self, observations):
        return [self.rng.randint(0, 4) for _ in observations]

    def act_batch(self, all_obs, positions):
        return [self.act(obs) for obs in all_obs]

    def reset_states(self):
        pass


class BatchAStarAgentWithActBatch(BatchAStarAgent):
    """Wraps BatchAStarAgent with act_batch for batched evaluation."""

    def __init__(self):
        super().__init__()
        self.per_env_agents = {}

    def act_batch(self, all_obs, positions):
        actions = []
        for obs, pos in zip(all_obs, positions):
            if pos not in self.per_env_agents:
                self.per_env_agents[pos] = {}
            env_agents = self.per_env_agents[pos]
            env_actions = []
            for idx, o in enumerate(obs):
                if idx not in env_agents:
                    env_agents[idx] = AStarAgent()
                env_actions.append(env_agents[idx].act(o))
            actions.append(env_actions)
        return actions

    def reset_states(self):
        super().reset_states()
        self.per_env_agents = {}


def test_smoke_eval(tmp_path):
    """Minimal end-to-end evaluation: registers env + A*, runs one episode, checks output."""
    ToolboxRegistry.register_env("Pogema-v0", create_env_base, Environment)
    ToolboxRegistry.register_algorithm("A*", BatchAStarAgent)

    eval_config = {
        "environment": {
            "name": "Pogema-v0",
            "observation_type": "POMAPF",
            "on_target": "restart",
            "max_episode_steps": 32,
            "num_agents": 2,
            "seed": 42,
            "map_size": 8,
            "density": 0.1,
        },
        "algorithms": {
            "A-Star": {
                "name": "A*",
                "num_process": 1,
                "parallel_backend": "sequential",
            },
        },
    }

    results = evaluation(eval_config, eval_dir=str(tmp_path))

    # Results list should be non-empty
    assert results, "evaluation() returned no results"

    # Each result dict must contain the expected top-level keys
    for r in results:
        assert "metrics" in r
        assert "env_grid_search" in r
        assert "algorithm" in r

    # JSON file should have been written for the algorithm
    json_path = tmp_path / "A-Star.json"
    assert json_path.exists(), f"Expected results file {json_path} not found"

    with open(json_path) as f:
        saved = json.load(f)
    assert isinstance(saved, list) and len(saved) > 0
    assert "metrics" in saved[0]


def test_batched_backend(tmp_path):
    """Runs evaluation with batched backend and verifies results match sequential."""
    ToolboxRegistry.register_env("Pogema-v0", create_env_base, Environment)
    ToolboxRegistry.register_algorithm("A*", BatchAStarAgentWithActBatch)

    base_env = {
        "name": "Pogema-v0",
        "observation_type": "POMAPF",
        "on_target": "restart",
        "max_episode_steps": 32,
        "num_agents": 2,
        "seed": {"grid_search": [42, 43]},
        "map_size": 8,
        "density": 0.1,
    }

    # Run sequential
    seq_results = evaluation(
        {
            "environment": base_env,
            "algorithms": {
                "A-Star-seq": {
                    "name": "A*",
                    "num_process": 1,
                    "parallel_backend": "sequential",
                },
            },
        },
        eval_dir=str(tmp_path / "seq"),
    )

    # Run batched
    batched_results = evaluation(
        {
            "environment": base_env,
            "algorithms": {
                "A-Star-batch": {
                    "name": "A*",
                    "num_process": 1,
                    "parallel_backend": "batched",
                    "batch_size": 4,
                },
            },
        },
        eval_dir=str(tmp_path / "batch"),
    )

    assert len(batched_results) == len(seq_results)

    for seq_r, batch_r in zip(seq_results, batched_results):
        assert seq_r["metrics"] == batch_r["metrics"]


def test_batched_random_agent(tmp_path):
    """Batched evaluation with a random agent produces valid results across multiple batches."""
    ToolboxRegistry.register_env("Pogema-v0", create_env_base, Environment)
    ToolboxRegistry.register_algorithm("Random", RandomAgent)

    results = evaluation(
        {
            "environment": {
                "name": "Pogema-v0",
                "observation_type": "POMAPF",
                "on_target": "restart",
                "max_episode_steps": 32,
                "num_agents": {"grid_search": [2, 4]},
                "seed": {"grid_search": [1, 2, 3]},
                "map_size": 8,
                "density": 0.1,
            },
            "algorithms": {
                "Random-batch": {
                    "name": "Random",
                    "num_process": 1,
                    "parallel_backend": "batched",
                    "batch_size": 4,
                },
            },
        },
        eval_dir=str(tmp_path),
    )

    assert len(results) == 6  # 2 num_agents x 3 seeds

    for r in results:
        assert "metrics" in r
        assert len(r["metrics"]) > 0
