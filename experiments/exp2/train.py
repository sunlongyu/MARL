# -*- coding: utf-8 -*-

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete, MultiDiscrete, Box, Dict as GymDictSpace
from typing import Dict as TypeDict, List, Optional
import functools

from pettingzoo import AECEnv
from pettingzoo.utils import wrappers, agent_selector
from pettingzoo.utils.conversions import parallel_wrapper_fn
from pettingzoo.test import api_test, parallel_api_test
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from typing import Tuple
from .env import raw_env
from .env import DEFENDER, ATTACKER, N_SYSTEMS_DEFAULT, MAX_STEPS_DEFAULT
from .config import *

def parallel_env(**kwargs):
    return parallel_wrapper_fn(lambda: raw_env(**kwargs))()

class TrainingTracker:
    def __init__(self):
        self.defender_rewards = []
        self.attacker_rewards = []
        self.defender_policy_loss = []
        self.attacker_policy_loss = []
        self.episodes = []
        self.epoch_results = []
        
    def update(self, iteration, result):
        env_info = result.get("env_runners", {})
        policy_rewards = env_info.get("policy_reward_mean", {})
        
        def_reward = policy_rewards.get("defender_policy", float('nan'))
        att_reward = policy_rewards.get("attacker_policy", float('nan')) 
        self.defender_rewards.append(def_reward)
        self.attacker_rewards.append(att_reward)
        
        metrics = result.get("info", {}).get("learner", {})
        def_policy_loss= metrics.get("defender_policy", {}).get("learner_stats",{}).get("policy_loss", float('nan'))
        att_policy_loss = metrics.get("attacker_policy", {}).get("learner_stats",{}).get("policy_loss", float('nan'))
        self.defender_policy_loss.append(def_policy_loss)
        self.attacker_policy_loss.append(att_policy_loss)
        
        self.episodes.append(iteration)
        self.epoch_results.append(result)

import os
import time
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO, PPOConfig
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.tune.registry import register_env
import torch
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from collections import defaultdict
import threading
import json

N_SYSTEMS=N_SYSTEMS_DEFAULT
MAX_STEPS=MAX_STEPS_DEFAULT

def create_rllib_env(config):
    """创建Ray兼容的环境包装器"""
    n_systems = config.get("n_systems", N_SYSTEMS)
    max_steps = config.get("max_steps", MAX_STEPS)
    history_length = config.get("history_length", 8)
    
    raw_parallel_env = parallel_env(N=n_systems, T=max_steps, history_length=history_length)
    env = ParallelPettingZooEnv(raw_parallel_env)
    
    print(f"\n观察空间信息:")
    for agent_id in env.observation_space:
        print(f"智能体 {agent_id} 的观察空间: {env.observation_space[agent_id]} (类型: {type(env.observation_space[agent_id]).__name__})")
    
    return env

def rllib_env_creator(env_config):
    return create_rllib_env(env_config)

def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id == DEFENDER:
        return "defender_policy"
    elif agent_id == ATTACKER:
        return "attacker_policy"
    else:
        raise ValueError(f"未知智能体ID: {agent_id}")

import abc
import json
import os

def save_training_results(results,epoch, dir_path="./epoch_results"):
    """保存训练结果到文件"""
    def convert_to_serializable(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (type, abc.ABCMeta)):
            return str(obj)
        if hasattr(obj, '__dict__'):
            return str(obj)
        if isinstance(obj, (set, frozenset)):
            return list(obj)
        return obj
    
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"epoch_{epoch}.json")
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4, default=convert_to_serializable)
    print(f"训练结果已保存至: {file_path}")

def save_training_results_by_filter(result_list, epoch, dir_path="./epoch_results"):
    """Save filtered training results focusing on key metrics"""
    
    def extract_key_metrics(result):
        info = result.get("info", None)
        if not info:
            return {}
        learner_metrics = info.get("learner", None)
        if learner_metrics is None:
            return {}
    
        defender_stats = {}
        if "defender_policy" in learner_metrics:
            defender_policy = learner_metrics.get("defender_policy", None)
            if "learner_stats" in defender_policy:
                defender_stats = learner_metrics.get("defender_policy", {}).get("learner_stats", {})
        
        attacker_stats = {}
        if "attacker_policy" in learner_metrics:
            attacker_policy = learner_metrics.get("attacker_policy", None)
            if "learner_stats" in attacker_policy:
                attacker_stats = learner_metrics.get("attacker_policy", {}).get("learner_stats", {})

        env_metrics = result.get("env_runners", {})
        
        metrics = {
            "training_iteration": result.get("training_iteration"),
            "timesteps_total": result.get("timesteps_total"),
            "time_total_s": result.get("time_total_s"),
            
            "info": {
                "defender_policy": {
                    stat: defender_stats.get(stat)
                    for stat in [
                        "grad_gnorm", "cur_kl_coeff", "cur_lr", "total_loss",
                        "policy_loss", "vf_loss", "vf_explained_var", "kl",
                        "entropy", "entropy_coeff"
                    ]
                },
                "attacker_policy": {    
                    stat: attacker_stats.get(stat)
                    for stat in [
                        "grad_gnorm", "cur_kl_coeff", "cur_lr", "total_loss",
                        "policy_loss", "vf_loss", "vf_explained_var", "kl",
                        "entropy", "entropy_coeff"
                    ]
                }
            },
            "env_runners": {
                "episode_reward_max": env_metrics.get("episode_reward_max"),
                "episode_reward_min": env_metrics.get("episode_reward_min"), 
                "episode_reward_mean": env_metrics.get("episode_reward_mean"),
                "episode_len_mean": env_metrics.get("episode_len_mean"),
                "policy_reward_min": {
                    "defender_policy": env_metrics.get("policy_reward_min", {}).get("defender_policy"),
                    "attacker_policy": env_metrics.get("policy_reward_min", {}).get("attacker_policy")
                },
                "policy_reward_max": {
                    "defender_policy": env_metrics.get("policy_reward_max", {}).get("defender_policy"),
                    "attacker_policy": env_metrics.get("policy_reward_max", {}).get("attacker_policy")
                },
                "policy_reward_mean": {
                    "defender_policy": env_metrics.get("policy_reward_mean", {}).get("defender_policy"),
                    "attacker_policy": env_metrics.get("policy_reward_mean", {}).get("attacker_policy")
                }
            }
        }
        
        return metrics

    def convert_to_serializable(obj):
        if isinstance(obj, (np.float32, np.float64)):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return obj

    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"epoch_{epoch}.json")
    
    filtered_results = [extract_key_metrics(result) for result in result_list]
    
    with open(file_path, 'w') as f:
        json.dump(filtered_results, f, indent=4, default=convert_to_serializable)

def save_training_results_by_interval(checkpoint_path,result_list, epoch, interval=10):
    """保存训练结果到文件，按指定轮次间隔"""
    dir_path = os.path.join(checkpoint_path, "epoch_results")
    if (epoch+1)!=1 and (epoch+1) % interval == 0:
        save_training_results_by_filter(result_list, epoch, dir_path)
        print(f"训练结果已保存至: {dir_path}/epoch_{epoch}.json")
        return True
    return False

def load_training_results(dir_path="./epoch_results"):
    """加载所有轮次的训练结果"""
    all_results = {}
    
    for filename in os.listdir(dir_path):
        if filename.startswith("epoch_") and filename.endswith(".json"):
            epoch = int(filename.split("_")[1].split(".")[0])
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as f:
                all_results[epoch] = json.load(f)
    
    sorted_results = dict(sorted(all_results.items()))
    return sorted_results

def save_config(config, file_path):
    """保存配置到JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存至: {file_path}")

def run_mappo_training(
    stop_iters=5,
    checkpoint_path="mappo_checkpoint",
    use_lstm=True,
    lstm_cell_size=64,
    seed=42,
    n_systems=N_SYSTEMS,
    max_steps=MAX_STEPS,
    train_batch_size=512,
    learning_rate=5e-4,
    defender_lr=2e-4,
    attacker_lr=1e-4,
    learning_rate_schedule=None,
    history_length=8,
    interval=10,
):
    """使用MAPPO训练防御者和攻击者策略"""
    print("初始化Ray...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,num_gpus=1,include_dashboard=False)
    print(f"Ray已初始化: {ray.is_initialized()}")
    print(f"Ray可用资源: {ray.available_resources()}")

    env_name = "multistage_signaling_exam2_v0"
    register_env(env_name, rllib_env_creator)

    temp_env = create_rllib_env({
        "n_systems": n_systems,
        "max_steps": max_steps,
        "history_length": history_length,
    })

    obs_spaces = temp_env.observation_space
    act_spaces = temp_env.action_space
    temp_env.close()

    config = (
        PPOConfig()
        .resources(num_gpus=1, num_cpus_per_worker=2)
        .environment(
            env=env_name,
            env_config={
                "n_systems": n_systems,
                "max_steps": max_steps,
                "history_length": history_length
            }
        )
        .framework("torch")
        .api_stack(
            enable_rl_module_and_learner=False,
            enable_env_runner_and_connector_v2=False,
        )
        .multi_agent(
            policies={
                "defender_policy": PolicySpec(
                    observation_space=obs_spaces[DEFENDER],
                    action_space=act_spaces[DEFENDER],
                    config={
                        "agent_id": DEFENDER,
                        "lr": defender_lr,
                    },
                ),
                "attacker_policy": PolicySpec(
                    observation_space=obs_spaces[ATTACKER],
                    action_space=act_spaces[ATTACKER],
                    config={
                        "agent_id": ATTACKER,
                        "lr": attacker_lr,
                    },
                ),
            },
            policy_mapping_fn=policy_mapping_fn,
            policies_to_train=["defender_policy", "attacker_policy"]
        )
        .update_from_dict({
            "batch_mode": "complete_episodes",
            "reward_clip": 5.0,
        })
        .training(
            model={
                "fcnet_hiddens": [lstm_cell_size, lstm_cell_size],
                "use_lstm": use_lstm,
                "lstm_cell_size": lstm_cell_size,
            },
            vf_clip_param=10.0,
            vf_loss_coeff=1.0,
            entropy_coeff=0.005,
            gamma=0.99,
            lambda_=0.97,
            num_sgd_iter=10,
            clip_param=0.2,
            lr=learning_rate,
            train_batch_size=train_batch_size,
        )
        .env_runners(
            num_env_runners=1,
            batch_mode="truncate_episodes",
            rollout_fragment_length=train_batch_size,
        )
        .debugging(
            log_level="ERROR", 
            log_sys_usage=True
        )
    )
    config.seed = seed
    
    algo = config.build()

    print("MAPPO算法创建成功!")

    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"检查点将保存至: {checkpoint_path}")

    print(f"开始训练，共{stop_iters}轮迭代...")
    start_time = time.time()

    train_results = []
    print("创建训练追踪器...") 
    tracker = TrainingTracker()
    
    result_list = []
    for i in range(stop_iters):
        iter_start = time.time()
        result = algo.train()
        result_list.append(result)
        if save_training_results_by_interval(checkpoint_path,result_list,i,interval=interval):
            result_list = []
        tracker.update(i, result)
        
        iter_time = time.time() - iter_start
        
        print(f"\n===== 迭代 {i+1}/{stop_iters}, 用时: {iter_time:.2f}秒 =====")
        checkpoint_interval = min(50, max(10, stop_iters // 10))
        if (i+1) % checkpoint_interval == 0 or (i+1) == stop_iters:
            checkpoint = algo.save(checkpoint_path)
            print(f"检查点已保存: {i}")

    final_checkpoint = algo.save(checkpoint_path)
    print(f"训练完成，总用时: {(time.time()-start_time)/60:.2f}分钟")
    print(f"最终检查点: {i}")
 
    algo.stop()
    if ray.is_initialized():
        ray.shutdown()

    return train_results, final_checkpoint, tracker

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TrainingVisualizer:
    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
        
    def plot_training_curves(self, tracker, learning_rate=0.0001):
        """绘制训练曲线"""
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        ax1.plot(tracker.episodes, tracker.defender_rewards, label='Defender', color='blue')
        ax1.plot(tracker.episodes, tracker.attacker_rewards, label='Attacker', color='red')
        
        window = 5
        def rolling_std(data, window):
            return pd.Series(data).rolling(window=window).std()
        
        def_std = rolling_std(tracker.defender_rewards, window)
        att_std = rolling_std(tracker.attacker_rewards, window)
        
        ax1.fill_between(tracker.episodes, 
                        [r - s for r, s in zip(tracker.defender_rewards, def_std)],
                        [r + s for r, s in zip(tracker.defender_rewards, def_std)],
                        alpha=0.2, color='blue')
        ax1.fill_between(tracker.episodes,
                        [r - s for r, s in zip(tracker.attacker_rewards, att_std)],
                        [r + s for r, s in zip(tracker.attacker_rewards, att_std)],
                        alpha=0.2, color='red')
        
        ax1.set_xlabel('Training Episodes')
        ax1.set_ylabel('Average Episode Reward')
        ax1.set_title(f'Training Progress: Rewards lr=({str(learning_rate)})')
        ax1.legend()
        ax1.grid(True)
        
        ax2.plot(tracker.episodes, tracker.defender_policy_loss, label='Defender', color='blue')
        ax2.plot(tracker.episodes, tracker.attacker_policy_loss, label='Attacker', color='red')
        ax2.set_xlabel('Training Episodes')
        ax2.set_ylabel('Policy Loss')  
        ax2.set_title(f'Training Progress: Policy Loss (lr={str(learning_rate)})')
        ax2.legend()
        ax2.grid(True)
        
        plt.tight_layout()
        plt.show()
        return fig
        
    def save_plots(self, tracker, save_path):
        """保存训练曲线图表"""
        fig = self.plot_training_curves(tracker)
        fig.savefig(save_path)
        plt.close(fig)

def train_and_save_results(
        stop_iters=200,
        use_lstm=True,
        lstm_cell_size=128,
        seed=0,
        n_systems=N_SYSTEMS,
        max_steps=MAX_STEPS,
        train_batch_size = 512,
        learning_rate=0.01,
        attacker_lr=1e-4,
        defender_lr=2e-4,  
        learning_rate_schedule=None,
        history_length=8,
        interval=10,
        checkpoint_dir_path="./checkpoint"
        ):
    """训练并保存结果"""
    checkpoint_dir_path = os.path.join(checkpoint_dir_path)
    os.makedirs(checkpoint_dir_path, exist_ok=True)

    env_config = {
        "n_systems": n_systems,
        "max_steps": max_steps,
        "history_length": history_length
    }
    with open(os.path.join(checkpoint_dir_path, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=4)
    print(f"环境配置已保存至: {os.path.join(checkpoint_dir_path, 'env_config.json')}")
    
    results, checkpoint, tracker = run_mappo_training(
        stop_iters=stop_iters,
        checkpoint_path=checkpoint_dir_path,
        use_lstm=use_lstm,
        lstm_cell_size=lstm_cell_size,
        seed=seed,
        n_systems=n_systems,
        max_steps=max_steps,
        train_batch_size = train_batch_size,
        learning_rate = learning_rate,
        attacker_lr = attacker_lr,
        defender_lr = defender_lr,
        history_length=history_length,
        interval=interval,
    )

    import pickle
    with open(checkpoint_dir_path+"/training_tracker.pkl", "wb") as f:
        pickle.dump(tracker, f)
 
    visualizer = TrainingVisualizer(figsize=(12, 10))
    visualizer.plot_training_curves(tracker,learning_rate=learning_rate)
    visualizer.save_plots(tracker, checkpoint_dir_path+"/training_curves.png")

def load_tracker_from_file(checkpoint_dir_path):
    """从文件加载训练跟踪器"""
    import pickle
    with open(checkpoint_dir_path+"/training_tracker.pkl", "rb") as f:
        tracker = pickle.load(f)
    return tracker

TRAINING_ITERATIONS = 400
USE_LSTM = True
LSTM_CELL_SIZE = 128
SEED = 58
CHECKPOINT_DIR_PATH = "checkpoints"

N_SYSTEMS = 4
MAX_STEPS = 25
HISTORY_LENGTH = 8
LEARNING_RATE = 1e-4
TRAIN_BATCH_SIZE = 128
LEARNING_RATE_SCHEDULE = None

if __name__ == "__main__":
    learn_rate_name = ["lr1", "lr2", "lr3","lr4"]
    learn_rate_list = [3e-4, 1.5e-4, 8e-5, 6e-5]
    attcker_lr_list = [3e-4, 1.5e-4, 8e-5, 6e-5]
    defender_lr_list = [3e-4, 1.5e-4, 8e-5, 6e-5]
    for learn_rate, name,index in zip(learn_rate_list, learn_rate_name,range(len(learn_rate_list))):
        print(f"开始训练，学习率: {learn_rate}")
        train_and_save_results(
            stop_iters=TRAINING_ITERATIONS,
            use_lstm=USE_LSTM,
            lstm_cell_size=LSTM_CELL_SIZE,
            seed=SEED,
            n_systems=N_SYSTEMS,
            max_steps=MAX_STEPS,
            learning_rate=learn_rate,
            attacker_lr = learn_rate,
            defender_lr = learn_rate,
            train_batch_size = TRAIN_BATCH_SIZE,
            learning_rate_schedule=LEARNING_RATE_SCHEDULE,
            history_length=HISTORY_LENGTH,
            interval=10,
            checkpoint_dir_path=f"{CHECKPOINT_DIR_PATH}_{name}"
        )
