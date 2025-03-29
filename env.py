# env.py
import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict as GymDictSpace # Use specific Dict name
from typing import Dict as TypeDict, List, Optional
import functools # For LRU cache and partial

# --- PettingZoo AECEnv ---
from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn
# 添加测试工具导入
from pettingzoo.test import api_test, parallel_api_test

# --- Version Check (Optional but recommended) ---
try:
    # Check for numpy version compatibility (Install numpy<2.0 if needed)
    np_version = tuple(map(int, np.__version__.split('.')[:2]))
    if np_version >= (2, 0):
        print(f"Warning: NumPy version {np.__version__} >= 2.0 detected. "
              "This might cause issues with older Ray/RLlib versions. "
              "Consider installing 'numpy<2.0'.")
except Exception as e:
    print(f"Warning: Could not check NumPy version: {e}")


# ==============================================================================
# === Constants and Payoffs (FROM YOUR VERSION - PLEASE DOUBLE-CHECK THESE!) ===
# ==============================================================================
DEFENDER = "defender"
ATTACKER = "attacker"
THETA_REAL = 0      # 真实系统类型
THETA_HONEYPOT = 1  # 蜜罐系统类型
SIGNAL_NORMAL = 0   # 正常信号 (s_N)
SIGNAL_HONEYPOT = 1 # 蜜罐信号 (s_H)
ACTION_RETREAT = 0  # 撤退 (a_R)
ACTION_ATTACK = 1   # 攻击 (a_A)

# --- Environment Parameters ---
N_SYSTEMS_DEFAULT = 3       # Default number of systems
MAX_STEPS_DEFAULT = 100     # Default max steps (T horizon)
HISTORY_LEN_DEFAULT = 5     # Default history length for observation

# --- Payoffs (CRITICAL: Verify these match your Table 1 exactly!) ---
R_A = 4.0  # Attacker gain from real system (θ1, sN, aA) -> U_A
C_A = 1.0   # Attacker cost of attack (subtracted when aA)
L_A = 5.0  # Defender loss from compromised real system (θ1, sN/sH, aA) -> U_D negative value
L_I = 3.0   # Defender intelligence gain / Attacker loss from honeypot (θ2, sN/sH, aA) -> U_D gain, U_A loss
C_N = 1.0   # Defender cost of deception on real system (θ1, sH) -> U_D cost
C_H = 2.0   # Defender cost of maintaining honeypot (θ2) -> U_D cost (When applied? Always? Check notes below)
A_R = 1.0  # Attacker EXTRA gain for (θ1, sH, aA)? (R_A - C_A + A_R)
D_R = 2.0  # Defender gain for (θ1, sH, aR)? (D_R - C_N)
A_D = 1.0  # Attacker gain for (θ2, sN/sH, aR)?

# ==============================================================================

# --- Environment Creation Functions ---
def env(render_mode=None, N=N_SYSTEMS_DEFAULT, T=MAX_STEPS_DEFAULT, history_length=HISTORY_LEN_DEFAULT):
    """Instantiates the AEC environment."""
    env = raw_env(render_mode=render_mode, N=N, T=T, history_length=history_length)
    # Apply standard wrappers
    env = wrappers.OrderEnforcingWrapper(env)
    return env

# 创建raw_env的函数版本，供parallel_wrapper_fn使用
def raw_env_fn(**kwargs):
    return raw_env(**kwargs)

# 使用正确的函数引用
parallel_env = functools.partial(parallel_wrapper_fn, raw_env_fn=raw_env_fn)

# --- Raw AEC Environment Class ---
class raw_env(AECEnv):
    """
    PettingZoo AEC environment for the Multi-Stage Signaling Game.

    Implements the logic based on the user's specification, including
    Dict observation spaces with history and specific payoff calculations.
    Designed for compatibility with RLlib PPO.
    """
    metadata = {
        "render_modes": ["human"],
        "name": "multistage_signaling_v2", # Incremented version
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
        self._agent_ids = set(self.possible_agents) # For quick checks
        self.render_mode = render_mode

        # --- Define Spaces (Using User's Dict Structure) ---
        self._defender_obs_space = GymDictSpace({
            # Normalized step count [0, 1]
            "current_step": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # True types known only to defender
            "system_types": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
            # Attacker's actions from the *previous* full step (t-1)
            "last_attacker_actions": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
            # History of attacker actions (most recent first)
            "attacker_action_history": Box(low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8),
            # History of own signals (most recent first)
            "defender_signal_history": Box(low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8),
        })
        self._attacker_obs_space = GymDictSpace({
            # Normalized step count [0, 1]
            "current_step": Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32),
            # Defender signals observed in the *current* step (t)
            "current_defender_signals": Box(low=0, high=1, shape=(self.N,), dtype=np.int8),
            # History of own actions (most recent first)
            "attacker_action_history": Box(low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8),
            # History of observed signals (most recent first)
            "defender_signal_history": Box(low=0, high=1, shape=(self.history_length, self.N), dtype=np.int8),
        })
        # Action space remains MultiDiscrete
        self._action_space = MultiDiscrete([2] * self.N)

        # --- Internal State Variables (Initialize) ---
        self.system_types: np.ndarray = np.zeros(self.N, dtype=np.int8)
        self.current_step: int = 0
        # Store signals/actions related to the current step being processed
        self._current_defender_signals: np.ndarray = np.zeros(self.N, dtype=np.int8)
        self._current_attacker_actions: np.ndarray = np.zeros(self.N, dtype=np.int8)
        # Store actions from the *previous* completed step for defender obs
        self._last_attacker_actions: np.ndarray = np.zeros(self.N, dtype=np.int8)
        # History buffers
        self._defender_signal_history: np.ndarray = np.zeros((self.history_length, self.N), dtype=np.int8)
        self._attacker_action_history: np.ndarray = np.zeros((self.history_length, self.N), dtype=np.int8)

        # --- AEC API Variables (Initialize) ---
        self.agents: List[str] = []
        self.rewards: TypeDict[str, float] = {}
        self._cumulative_rewards: TypeDict[str, float] = {}
        self.terminations: TypeDict[str, bool] = {}
        self.truncations: TypeDict[str, bool] = {}
        self.infos: TypeDict[str, TypeDict] = {}
        self._agent_selector = agent_selector(self.possible_agents)
        self.agent_selection: str = "" # Will be set in reset

    # --- PettingZoo API Properties ---
    @functools.lru_cache(maxsize=None)
    def observation_space(self, agent: str) -> gym.spaces.Space:
        if agent == DEFENDER: return self._defender_obs_space
        if agent == ATTACKER: return self._attacker_obs_space
        raise ValueError(f"Unknown agent: {agent}")

    @functools.lru_cache(maxsize=None)
    def action_space(self, agent: str) -> gym.spaces.Space:
        # Both agents have the same action space structure in this case
        return self._action_space

    # --- Core Methods ---
    def reset(self, seed: Optional[int] = None, options: Optional[dict] = None) -> None:
        """Resets the environment to a starting state."""
        if seed is not None:
            np.random.seed(seed)

        # --- Reset Internal State ---
        # Example: Random types, ensure dtype for consistency
        self.system_types = np.random.randint(0, 2, size=self.N, dtype=np.int8)
        # Example: Fixed types for testing
        # self.system_types = np.array([THETA_REAL, THETA_HONEYPOT, THETA_HONEYPOT], dtype=np.int8)
        # np.random.shuffle(self.system_types)

        self.current_step = 0
        self._current_defender_signals.fill(0)
        self._current_attacker_actions.fill(0)
        self._last_attacker_actions.fill(0) # Reset for first defender obs
        self._defender_signal_history.fill(0)
        self._attacker_action_history.fill(0)

        # --- Reset AEC API Variables ---
        self.agents = list(self.possible_agents) # Use a fresh list copy
        self._agent_selector.reinit(self.agents)
        self.agent_selection = self._agent_selector.reset() # Get the first agent (Defender)

        self.rewards = {agent: 0.0 for agent in self.agents}
        self._cumulative_rewards = {agent: 0.0 for agent in self.agents}
        self.terminations = {agent: False for agent in self.agents}
        self.truncations = {agent: False for agent in self.agents}
        # Provide initial info, including true types for defender
        self.infos = {agent: {"step": self.current_step} for agent in self.agents}
        self.infos[DEFENDER]["true_types"] = self.system_types.copy() # Provide a copy

        # --- Render initial state if requested ---
        if self.render_mode == "human":
            self.render()

        # NOTE: AEC reset() does not return observations. The caller (e.g., RLlib)
        # will call last() or observe() after reset() to get the first observation.

    def observe(self, agent: str) -> TypeDict[str, np.ndarray]:
        """Returns the observation for the specified agent."""
        if agent not in self._agent_ids:
             raise ValueError(f"observe() called for invalid agent: {agent}")

        normalized_step = np.array([self.current_step / self.T], dtype=np.float32)

        if agent == DEFENDER:
            # Defender observes previous attacker actions and own history
            obs = {
                "current_step": normalized_step,
                "system_types": self.system_types.copy(), # Knows true types
                "last_attacker_actions": self._last_attacker_actions.copy(), # Actions from t-1
                "attacker_action_history": self._attacker_action_history.copy(),
                "defender_signal_history": self._defender_signal_history.copy(),
            }
        elif agent == ATTACKER:
            # Attacker observes current defender signals and own history
            obs = {
                "current_step": normalized_step,
                "current_defender_signals": self._current_defender_signals.copy(), # Signals from t
                "attacker_action_history": self._attacker_action_history.copy(),
                "defender_signal_history": self._defender_signal_history.copy(),
            }
        else:
             # Should be unreachable due to initial check
             raise ValueError(f"Unknown agent type for observation: {agent}")

        # Ensure all numpy arrays in the dict have the correct dtype if needed,
        # although RLlib's wrappers usually handle this.
        return obs

    def state(self) -> np.ndarray:
        """Returns a global state view (e.g., for centralized critic)."""
        # Flatten relevant parts of the state into a single vector
        # Note: History buffers might make this large. Consider alternatives if needed.
        flat_state = np.concatenate([
            self.system_types.astype(np.float32),
            np.array([self.current_step / self.T], dtype=np.float32),
            self._current_defender_signals.astype(np.float32),
            self._last_attacker_actions.astype(np.float32), # Use last actions maybe? Or current?
            self._defender_signal_history.flatten().astype(np.float32),
            self._attacker_action_history.flatten().astype(np.float32),
        ])
        return flat_state

    def step(self, action: np.ndarray) -> None:
        """Performs a step for the current agent_selection."""
        if self.terminations.get(self.agent_selection, False) or self.truncations.get(self.agent_selection, False):
            # Agent is done, handle illegal step (PettingZoo standard practice)
            self._was_dead_step(action)
            return

        agent = self.agent_selection
        # Ensure action is numpy array with correct dtype
        action = np.asarray(action, dtype=self._action_space.dtype)

        # --- Store Action & Update History ---
        if agent == DEFENDER:
            self._current_defender_signals = action # Store signals for current step t
            # Update history (roll and add to front)
            self._defender_signal_history = np.roll(self._defender_signal_history, shift=1, axis=0)
            self._defender_signal_history[0, :] = action
        elif agent == ATTACKER:
            self._current_attacker_actions = action # Store actions for current step t
            # Update history
            self._attacker_action_history = np.roll(self._attacker_action_history, shift=1, axis=0)
            self._attacker_action_history[0, :] = action
        else:
             # Should be unreachable
             raise ValueError(f"step() called for invalid agent: {agent}")

        # --- Environment Transition & Rewards (only after the LAST agent acts) ---
        if self._agent_selector.is_last():
            # Attacker was the last agent, now the environment state advances

            # 1. Calculate Rewards for step t based on D's signals and A's actions
            step_rewards = self._calculate_rewards(
                self._current_defender_signals,
                self._current_attacker_actions
            )
            # Assign rewards for this step
            self.rewards[DEFENDER] = step_rewards[DEFENDER]
            self.rewards[ATTACKER] = step_rewards[ATTACKER]

            # 2. Accumulate total rewards for the episode
            self._cumulative_rewards[DEFENDER] += step_rewards[DEFENDER]
            self._cumulative_rewards[ATTACKER] += step_rewards[ATTACKER]

            # 3. Advance timestep
            self.current_step += 1

            # 4. Update _last_attacker_actions (needed for *next* defender observation)
            self._last_attacker_actions = self._current_attacker_actions.copy()

            # 5. Check for episode end (Truncation due to time limit)
            is_truncated = self.current_step >= self.T
            self.truncations = {a: is_truncated for a in self.agents}
            # Set terminations based on truncations for simplicity here,
            # can add other termination conditions if needed.
            self.terminations = {a: is_truncated for a in self.agents}

            # 6. Update infos for all agents after the step
            self.infos = {a: {"step": self.current_step} for a in self.agents}
            # Re-add persistent info like true types for defender
            self.infos[DEFENDER]["true_types"] = self.system_types.copy()

            # 7. Reset agents list based on terminations/truncations for next iter()
            # self.agents = [a for a in self.agents if not (self.terminations[a] or self.truncations[a])]
            # Let agent_selector handle this based on flags

        else:
            # Not the last agent (Defender just moved)
            # Clear the rewards dict for the current step; rewards are assigned later
            self._clear_rewards()

        # --- Select next agent ---
        # This needs to happen regardless of whether it was the last agent
        # It prepares for the *next* call to step() or last()
        self.agent_selection = self._agent_selector.next()

        # --- Add rewards to agent's cumulative reward (AEC API standard) ---
        # This happens *after* the agent acts but *before* the next agent starts
        self._accumulate_rewards() # Updates _cumulative_rewards based on self.rewards set above

        # --- Render after step logic if needed ---
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

            # --- Payoff Logic (CRITICAL: VERIFY THIS SECTION) ---
            # Comments indicate the scenario (Type, Signal, Action) -> (Defender Utility, Attacker Utility)
            if theta_i == THETA_REAL:
                if s_i == SIGNAL_NORMAL:
                    if a_i == ACTION_ATTACK: # (θ1, sN, aA)
                        u_d, u_a = -L_A, R_A - C_A
                    else:                   # (θ1, sN, aR)
                        u_d, u_a = 0.0, 0.0
                else: # SIGNAL_HONEYPOT
                    cost_n = C_N # Deception cost
                    if a_i == ACTION_ATTACK: # (θ1, sH, aA)
                        u_d, u_a = -L_A - cost_n, R_A - C_A + A_R # Added A_R based on your constants
                    else:                   # (θ1, sH, aR)
                        u_d, u_a = D_R - cost_n, 0.0 # Added D_R based on your constants
            else: # THETA_HONEYPOT
                # Apply honeypot maintenance cost C_H?
                # Assumption: Applied *always* if it's a honeypot. Verify this!
                cost_h = C_H
                if s_i == SIGNAL_NORMAL:
                    if a_i == ACTION_ATTACK: # (θ2, sN, aA)
                        u_d, u_a = L_I - cost_h, -C_A - L_I
                    else:                   # (θ2, sN, aR)
                        u_d, u_a = -cost_h, A_D # Added A_D based on your constants
                else: # SIGNAL_HONEYPOT
                    if a_i == ACTION_ATTACK: # (θ2, sH, aA)
                        # Your original calculation: u_d = L_I. Did C_H still apply?
                        # Assuming C_H still applies:
                        u_d, u_a = L_I - cost_h, -C_A - L_I
                    else:                   # (θ2, sH, aR)
                        # Your original calculation: u_d = 0. Did C_H still apply?
                        # Assuming C_H still applies:
                        u_d, u_a = -cost_h, A_D # Added A_D based on your constants
            # --- END VERIFICATION SECTION ---

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
            # Optional: Print recent history snippet
            print("--- History (Last 3) ---")
            print(" D Signals : ")
            print(self._defender_signal_history[:min(3, self.history_length)])
            print(" A Actions : ")
            print(self._attacker_action_history[:min(3, self.history_length)])
            print("-" * 40)

    def close(self):
        """Closes the environment (no-op here)."""
        pass


# ==============================================================================
# === Test Code (Run this file directly: python env.py) ===
# ==============================================================================
if __name__ == "__main__":
    import traceback

    # --- Parameters for Testing ---
    test_N = 3
    test_T = 15
    test_hist_len = 4

    # --- Test AEC Environment ---
    print("\n--- Running AEC API test ---")
    aec_env_instance = None
    try:
        # Pass parameters to env creator
        aec_env_instance = env(N=test_N, T=test_T, history_length=test_hist_len)
        api_test(aec_env_instance, num_cycles=1000, verbose_progress=False)
        print(f"AEC API test PASSED for {aec_env_instance.metadata['name']}")
    except Exception as e:
        print(f"AEC API test FAILED: {e}")
        traceback.print_exc()
    finally:
        if aec_env_instance: aec_env_instance.close()

    # --- Test Parallel Environment ---
    print("\n--- Running Parallel API test ---")
    par_env_instance = None
    try:
        # Create parallel env using the partial function with parameters
        par_env_creator = functools.partial(parallel_env, N=test_N, T=test_T, history_length=test_hist_len)
        # The test function expects a callable that returns an env
        parallel_api_test(par_env_creator, num_cycles=1000)
        print("Parallel API test PASSED.")
    except Exception as e:
        print(f"Parallel API test FAILED: {e}")
        traceback.print_exc()
    # parallel_api_test doesn't return the env instance for closing

    # --- Manual Interaction Example ---
    print("\n--- Manual Interaction Example ---")
    manual_env = None
    try:
        manual_env = env(render_mode="human", N=test_N, T=test_T, history_length=test_hist_len)
        manual_env.reset(seed=42)
        step_count = 0
        # Use agent_iter for clearer loop control
        for agent in manual_env.agent_iter():
            observation, reward, termination, truncation, info = manual_env.last()

            print(f"\nAgent: {agent}, Step: {manual_env.current_step}")
            # print(f"Obs: {observation}") # Can be very verbose
            print(f"Reward received by {agent}: {reward:.2f}")
            print(f"Done flags: term={termination}, trunc={truncation}")

            if termination or truncation:
                action = None # Agent is done
                print(f"Action for {agent}: None (Terminated/Truncated)")
            else:
                action = manual_env.action_space(agent).sample() # Sample random action
                print(f"Action for {agent}: {action}")

            manual_env.step(action)
            step_count += 1
            if step_count > (test_T + 2) * len(manual_env.possible_agents): # Safety break
                 print("Warning: Manual loop exceeded expected steps.")
                 break
        print("\nManual Interaction Finished.")
    except Exception as e:
        print(f"Manual Interaction FAILED: {e}")
        traceback.print_exc()
    finally:
        if manual_env: manual_env.close()