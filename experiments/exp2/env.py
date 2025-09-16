# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict as GymDictSpace
from typing import Dict as TypeDict, List, Optional
import functools

# --- PettingZoo AECEnv ---
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.test import api_test, parallel_api_test

from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv

from typing import Tuple

from .config import *

# --- Environment Creation Functions ---
def env(render_mode=None, N=N_SYSTEMS_DEFAULT, T=MAX_STEPS_DEFAULT, history_length=HISTORY_LEN_DEFAULT):
    """Instantiates the AEC environment."""
    env = raw_env(render_mode=render_mode, N=N, T=T, history_length=history_length)
    env = wrappers.OrderEnforcingWrapper(env)
    return env

def raw_env_fn(**kwargs):
    return raw_env(**kwargs)

def parallel_env(**kwargs):
    return parallel_wrapper_fn(lambda: raw_env(**kwargs))()

# --- Raw AEC Environment Class ---
class raw_env(AECEnv):
    """
    PettingZoo AEC environment for the Multi-Stage Signaling Game - Exam2 variant.
    """
    metadata = {
        "render_modes": ["human"],
        "name": "multistage_signaling_exam2_v1",
        "is_parallelizable": True,
        "has_manual_policy": False,
    }

    def __init__(self, render_mode=None, N=N_SYSTEMS_DEFAULT, T=MAX_STEPS_DEFAULT, history_length=HISTORY_LEN_DEFAULT):
        super().__init__()

        # --- Store Parameters ---
        self.N = N
        self.T = T
        if history_length <= 0:
             print("Warning: history_length should be positive. Setting to 1.")
             history_length = 1
        self.history_length = history_length
        self.possible_agents = [DEFENDER, ATTACKER]
        self.agent_ids = self.possible_agents
        self._agent_ids = set(self.possible_agents)
        self.render_mode = render_mode

        # --- Define Spaces ---
        self._defender_obs_space = GymDictSpace({
            "current_step": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "system_types": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
            "last_attacker_actions": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
            "attacker_action_history": Box(low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8),
            "defender_signal_history": Box(low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8),
        })
        self._attacker_obs_space = GymDictSpace({
            "current_step": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            "current_defender_signals": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
            "attacker_action_history": Box(low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8),
            "defender_signal_history": Box(low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8),
            "belief_real": Box(low=0.0, high=1.0, shape=(self.N,), dtype=np.float32),
        })
        self._action_space = MultiDiscrete([2] * self.N)

        # 信念更新参数
        self.P_theta = [0.6, 0.4]  # 先验概率
        self.lambda_D = [[0.75, 0.25], [0.25, 0.75]]  # 似然函数
        self.beta = 0.12  # 衰减因子

        # --- Internal State Variables ---
        self.system_types: np.ndarray = np.zeros(self.N, dtype=np.int8)
        self.current_step: int = 0
        self._current_defender_signals: np.ndarray = np.zeros(self.N, dtype=np.int8)
        self._current_attacker_actions: np.ndarray = np.zeros(self.N, dtype=np.int8)
        self._last_attacker_actions: np.ndarray = np.zeros(self.N, dtype=np.int8)
        self._defender_signal_history: np.ndarray = np.zeros((self.history_length, self.N), dtype=np.int8)
        self._attacker_action_history: np.ndarray = np.zeros((self.history_length, self.N), dtype=np.int8)

        # --- AEC API Variables ---
        self.agents: List[str] = []
        self.rewards: TypeDict[str, float] = {}
        self._cumulative_rewards: TypeDict[str, float] = {}
        self.terminations: TypeDict[str, bool] = {}
        self.truncations: TypeDict[str, bool] = {}
        self.infos: TypeDict[str, TypeDict] = {}
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection: str = ""

    # --- PettingZoo API Properties ---
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.spaces.Space:
        if agent == DEFENDER: return self._defender_obs_space
        if agent == ATTACKER: return self._attacker_obs_space
        raise ValueError(f"Unknown agent: {agent}")

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Space:
        return self._action_space

    # --- Core Methods ---
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """Resets the environment to a starting state."""
        if seed is not None:
            np.random.seed(seed)

        # --- Reset Internal State ---
        self.system_types = np.random.randint(0, 2, size=self.N, dtype=np.int8)
        self.current_step = 0
        self._current_defender_signals.fill(0)
        self._current_attacker_actions.fill(0)
        self._last_attacker_actions.fill(0)
        self._defender_signal_history.fill(0)
        self._attacker_action_history.fill(0)
        self._full_signal_history = [[] for _ in range(self.N)]

        # --- Reset AEC API Variables ---
        self.agents = list(self.possible_agents)
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset()

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        self.infos = {agent: {"step": self.current_step} for agent in self.agents}
        self.infos[DEFENDER]["true_types"] = self.system_types.copy()

        if self.render_mode == "human":
            self.render()

        self.rewards = {DEFENDER: 0.0, ATTACKER: 0.0}
        self._cumulative_rewards = {DEFENDER: 0.0, ATTACKER: 0.0}
        self.has_reset = True

    def observe(self, agent: str) -> np.ndarray:
        """Returns the observation for the specified agent."""
        if agent not in self.agents:
            print(f"[DEBUG][OBSERVE] {agent} 不在 agents 中，无法返回 obs")
            return self.observation_space(agent).sample() * 0
        if agent not in self._agent_ids:
             raise ValueError(f"observe() called for invalid agent: {agent}")

        normalized_step = np.array([self.current_step / self.T], dtype=np.float32)

        if agent == DEFENDER:
            obs = {
                "current_step": normalized_step,
                "system_types": self.system_types.copy(),
                "last_attacker_actions": self._last_attacker_actions.copy(),
                "attacker_action_history": self._attacker_action_history.copy(),
                "defender_signal_history": self._defender_signal_history.copy(),
            }
        elif agent == ATTACKER:
            belief_real = np.array([
                self._compute_belief(i) for i in range(self.N)
            ], dtype=np.float32)
            obs = {
                "current_step": normalized_step,
                "current_defender_signals": self._current_defender_signals.copy(),
                "attacker_action_history": self._attacker_action_history.copy(),
                "defender_signal_history": self._defender_signal_history.copy(),
                "belief_real": belief_real,
            }
        else:
             raise ValueError(f"Unknown agent type for observation: {agent}")

        for k, v in obs.items():
          if np.isnan(v).any():
              print(f"[NaN] Key={k}, Step={self.current_step}, Agent={agent}")

        return obs

    def state(self) -> np.ndarray:
        """Returns a global state view."""
        flat_state = np.concatenate([
            self.system_types.astype(np.float32),
            np.array([self.current_step / self.T], dtype=np.float32),
            self._current_defender_signals.astype(np.float32),
            self._last_attacker_actions.astype(np.float32),
            self._defender_signal_history.flatten().astype(np.float32),
            self._attacker_action_history.flatten().astype(np.float32),
        ])
        return flat_state

    def _compute_belief(self, system_idx: int) -> float:
        """计算系统 i 为真实系统的信念"""
        signals = self._full_signal_history[system_idx]
        if not signals:
            return 0.6  # 返回先验概率

        t = len(signals)
        log_psi_tilde = np.zeros(2)

        for theta in [0, 1]:
            log_psi_tilde[theta] = np.log(self.P_theta[theta])
            for t_prime, s in enumerate(signals, start=1):
                log_psi_tilde[theta] += np.log(self.lambda_D[theta][s]) - self.beta * (t - t_prime)

        max_log = np.max(log_psi_tilde)
        sum_exp = np.sum(np.exp(log_psi_tilde - max_log))
        log_Z = max_log + np.log(sum_exp)
        psi = np.exp(log_psi_tilde - log_Z)

        return psi[0]

    def step(self, action: np.ndarray) -> None:
        """Performs a step for the current agent_selection."""
        if self.terminations.get(self.agent_selection, False) or self.truncations.get(self.agent_selection, False):
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        action = np.asarray(action, dtype=self._action_space.dtype)

        # --- Store Action & Update History ---
        if agent == DEFENDER:
            self._current_defender_signals = action
            self._defender_signal_history = np.roll(self._defender_signal_history, shift=1, axis=0)
            self._defender_signal_history[0, :] = action
            for i in range(self.N):
                self._full_signal_history[i].append(int(self._current_defender_signals[i]))
        elif agent == ATTACKER:
            self._current_attacker_actions = action
            self._attacker_action_history = np.roll(self._attacker_action_history, shift=1, axis=0)
            self._attacker_action_history[0, :] = action
        else:
             raise ValueError(f"step() called for invalid agent: {agent}")

        # --- Environment Transition & Rewards ---
        if self._agent_selector.is_last():
            step_rewards = self._calculate_rewards(
                self._current_defender_signals,
                self._current_attacker_actions
            )
            self.rewards[DEFENDER] = step_rewards[DEFENDER]
            self.rewards[ATTACKER] = step_rewards[ATTACKER]

            self._cumulative_rewards[DEFENDER] += step_rewards[DEFENDER]
            self._cumulative_rewards[ATTACKER] += step_rewards[ATTACKER]

            self.current_step += 1
            self._last_attacker_actions = self._current_attacker_actions.copy()

            is_truncated = self.current_step >= self.T
            self.truncations = {a: is_truncated for a in self.agents}
            self.terminations = {a: is_truncated for a in self.agents}

            self.infos = {a: {"step": self.current_step} for a in self.agents}
            self.infos[DEFENDER]["true_types"] = self.system_types.copy()

        self.agent_selection = self._agent_selector.next()

        self._accumulate_rewards()
        if all(self.terminations.get(agent, False) or self.truncations.get(agent, False) for agent in self.agents):
           self.agents = []
           self.infos = {}

        if self.render_mode == "human":
            self.render()

    def _calculate_rewards(self, defender_signals_t: np.ndarray, attacker_actions_t: np.ndarray) -> TypeDict[str, float]:
        """Calculates rewards based on types, signals, and actions."""
        total_reward_D = 0.0
        total_reward_A = 0.0

        for i in range(self.N):
            theta_i = self.system_types[i]
            s_i = defender_signals_t[i]
            a_i = attacker_actions_t[i]

            u_d, u_a = 0.0, 0.0

            if theta_i == THETA_REAL:
                if s_i == SIGNAL_NORMAL:
                    if a_i == ACTION_ATTACK:
                        u_d, u_a = -L_A, R_A - C_A
                    else:
                        u_d, u_a = 0.0, 0.0
                else:
                    if a_i == ACTION_ATTACK:
                        u_d, u_a = -L_A - C_N, R_A - C_A + A_R
                    else:
                        u_d, u_a = D_R - C_N, 0.0
            else:
                if s_i == SIGNAL_NORMAL:
                    if a_i == ACTION_ATTACK:
                        u_d, u_a = L_I - C_H, -C_A - L_I
                    else:
                        u_d, u_a = -C_H, A_D
                else:
                    if a_i == ACTION_ATTACK:
                        u_d, u_a = L_I - C_H, -C_A - L_I
                    else:
                        u_d, u_a = -C_H, A_D

            total_reward_D += u_d
            total_reward_A += u_a

        return {DEFENDER: total_reward_D, ATTACKER: total_reward_A}

    def render(self):
        """Renders the environment state."""
        if self.render_mode == "human":
            print("-" * 40)
            print(f"Step: {self.current_step} / {self.T}")
            print(f"System Types : {self.system_types}")
            print(f"Defender Sigs: {self._current_defender_signals}")
            print(f"Attacker Acts: {self._current_attacker_actions} (Last: {self._last_attacker_actions})")
            print(f"Step Rewards : Def={self.rewards.get(DEFENDER, 0.0):.2f}, Att={self.rewards.get(ATTACKER, 0.0):.2f}")
            print(f"Total Rewards: Def={self._cumulative_rewards.get(DEFENDER, 0.0):.2f}, Att={self._cumulative_rewards.get(ATTACKER, 0.0):.2f}")
            print(f"Agent to Act : {self.agent_selection}")
            print("-" * 40)

    def last(self) -> Tuple[np.ndarray, float, bool, bool, dict]:
        agent = self.agent_selection
        obs = self.observe(agent)
        reward = self.rewards.get(agent, 0.0)
        terminated = self.terminations.get(agent, False)
        truncated = self.truncations.get(agent, False)
        info = self.infos.get(agent, {})

        return obs, reward, terminated, truncated, info

    def close(self):
        """Closes the environment."""
        pass
