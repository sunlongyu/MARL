# -*- coding: utf-8 -*-

# !pip install -U ray

# !pip install lz4

# !pip install gymnasium

# !pip install pettingzoo

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
from ray.rllib.env.wrappers.pettingzoo_env import ParallelPettingZooEnv
from typing import Tuple
from experiments.exam1.env import raw_env
from experiments.exam1.env import DEFENDER, ATTACKER, N_SYSTEMS_DEFAULT, MAX_STEPS_DEFAULT
from experiments.exam1.config import *


def parallel_env(**kwargs):
    return parallel_wrapper_fn(lambda: raw_env(**kwargs))()


#
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
        
        # 记录奖励
        def_reward = policy_rewards.get("defender_policy", float('nan'))
        att_reward = policy_rewards.get("attacker_policy", float('nan')) 
        self.defender_rewards.append(def_reward)
        self.attacker_rewards.append(att_reward)
        
        # 记录策略损失
        metrics = result.get("info", {}).get("learner", {})
        def_policy_loss= metrics.get("defender_policy", {}).get("learner_stats",{}).get("policy_loss", float('nan'))
        att_policy_loss = metrics.get("attacker_policy", {}).get("learner_stats",{}).get("policy_loss", float('nan'))
        self.defender_policy_loss.append(def_policy_loss)
        self.attacker_policy_loss.append(att_policy_loss)
        
        self.episodes.append(iteration)
        self.epoch_results.append(result)
# --- Training Functions ---

import os
import time
import numpy as np
import ray
# from ray.rllib.algorithms.ppo import PPO
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
# 修改环境创建函数，使用跟踪器
def create_rllib_env(config):
    """创建Ray兼容的环境包装器，包含跟踪器功能"""
    # 从config中获取参数
    n_systems = config.get("n_systems", N_SYSTEMS)
    max_steps = config.get("max_steps", MAX_STEPS)
    history_length = config.get("history_length", 5)
    use_tracker = config.get("use_tracker", False)  # 是否使用跟踪器
    
    
    # 不使用跟踪器，直接创建parallel环境
    raw_parallel_env = parallel_env(N=n_systems, T=max_steps, history_length=history_length)
    
    # 包装为Ray兼容的环境
    env = ParallelPettingZooEnv(raw_parallel_env)
    
        
    # 打印观察空间信息
    print(f"\n观察空间信息:")
    for agent_id in env.observation_space:
        print(f"智能体 {agent_id} 的观察空间: {env.observation_space[agent_id]} (类型: {type(env.observation_space[agent_id]).__name__})")
    
    return env

# 修改环境创建器函数，传递跟踪器配置
def rllib_env_creator(env_config):
    return create_rllib_env(env_config)

# 策略映射函数
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
#保存训练轮次结果信息
def save_training_results(results,epoch, dir_path="./epoch_results"):
    """保存训练结果到文件"""
    def convert_to_serializable(obj):
        if isinstance(obj, np.float32):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (type, abc.ABCMeta)):  # Handle metaclasses
            return str(obj)
        if hasattr(obj, '__dict__'):  # Handle custom objects
            return str(obj)
        if isinstance(obj, (set, frozenset)):  # Handle sets
            return list(obj)
        return obj
    
    os.makedirs(dir_path, exist_ok=True)
    file_path = os.path.join(dir_path, f"epoch_{epoch}.json")
    with open(file_path, 'w') as f:
        json.dump(results, f, indent=4, default=convert_to_serializable)
    print(f"训练结果已保存至: {file_path}")

# Define the paths to save with nested structure
SAVE_NAME_MAP = {
    "info": {
        "learner": {
            "defender_policy": {
                "learner_stats": ["grad_gnorm", "cur_kl_coeff", "cur_lr", "total_loss", 
                                "policy_loss", "vf_loss", "vf_explained_var", "kl", 
                                "entropy", "entropy_coeff"]
            },
            "attacker_policy": {
                "learner_stats": ["grad_gnorm", "cur_kl_coeff", "cur_lr", "total_loss",
                                "policy_loss", "vf_loss", "vf_explained_var", "kl",
                                "entropy", "entropy_coeff"] 
            }
        }
    },
    "env_runners": {
        "episode_reward_max": None,
        "episode_reward_min": None,
        "episode_reward_mean": None,
        "episode_len_mean": None,
        "policy_reward_min": ["defender_policy", "attacker_policy"],
        "policy_reward_max": ["defender_policy", "attacker_policy"],
        "policy_reward_mean": ["defender_policy", "attacker_policy"]
    }
}
"""
1. 梯度相关
grad_gnorm: 梯度的全局范数
衡量策略更新的幅度
较大值表示更新剧烈
较小值表示更新稳定

2. KL散度相关
cur_kl_coeff: 当前KL散度系数
用于控制策略更新的保守程度
较大值会使策略更新更保守
自适应调整以保持策略稳定性

kl: KL散度值
衡量新旧策略分布的差异
较小值表示更新保守
较大值表示更新激进
理想情况：
    应保持在0.01-0.02之间
    如果太小(<0.01)说明策略几乎不更新
    如果太大(>0.05)说明更新不稳定
KL散度稳定之后代表策略拟合

3. 学习率相关
cur_lr: 当前学习率
控制策略更新步长
可能会随训练进行衰减
影响训练的稳定性和收敛速度

4. 损失函数相关
total_loss: 总体损失
策略损失和值函数损失的加权和
反映整体训练效果

policy_loss: 策略损失
衡量策略改进的程度
PPO的核心优化目标

vf_loss: 值函数损失
状态值估计的准确程度
影响优势估计的质量

5. 值函数相关
vf_explained_var: 值函数解释方差
范围[-1,1]
越接近1表示值函数预测越准确
负值表示预测效果差于平均基线
理想情况：
    应该逐渐增大并稳定在0.1以上
    当前值较小，说明值函数拟合还不够好


6. 熵相关
entropy: 策略熵
衡量策略的探索程度
较高值表示更多探索
较低值表示更多利用
理想情况：应该逐渐减小并稳定

entropy_coeff: 熵系数
控制探索-利用平衡
较大值鼓励探索
通常随训练进程递减

"""
def save_training_results_by_filter(result_list, epoch, dir_path="./epoch_results"):
    """Save filtered training results focusing on key metrics"""
    
    def extract_key_metrics(result):
        # Extract info.learner metrics
        info = result.get("info", None)
        if not info:
            return {}
        learner_metrics = info.get("learner", None)
        if learner_metrics is None:
            return {}
    
        defender_stats = {}
        if "defender_policy" in learner_metrics:
            # Extract defender policy metrics
            defender_policy = learner_metrics.get("defender_policy", None)
            if "learner_stats" in defender_policy:
                defender_stats = learner_metrics.get("defender_policy", {}).get("learner_stats", {})
        
        attacker_stats = {}
        if "attacker_policy" in learner_metrics:
            # Extract defender policy metrics
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
    
    # Extract key metrics from results
    filtered_results = [extract_key_metrics(result) for result in result_list]
    
    with open(file_path, 'w') as f:
        json.dump(filtered_results, f, indent=4, default=convert_to_serializable)
    
#按制定轮次间隔保存
def save_training_results_by_interval(checkpoint_path,result_list, epoch, interval=10):
    """保存训练结果到文件，按指定轮次间隔"""
    dir_path = os.path.join(checkpoint_path, "epoch_results")
    if (epoch+1)!=1 and (epoch+1) % interval == 0:
        save_training_results_by_filter(result_list, epoch, dir_path)
        print(f"训练结果已保存至: {dir_path}/epoch_{epoch}.json")
        return True
    return False



def load_training_results(dir_path="./epoch_results"):
    """加载所有轮次的训练结果
    
    Args:
        dir_path (str): 结果文件存储目录路径
        
    Returns:
        dict: 包含所有轮次训练结果的字典，key为轮次编号，value为对应结果
    """
    all_results = {}
    
    # 遍历目录下所有文件
    for filename in os.listdir(dir_path):
        if filename.startswith("epoch_") and filename.endswith(".json"):
            # 提取轮次编号
            epoch = int(filename.split("_")[1].split(".")[0])
            # 加载该轮次结果
            file_path = os.path.join(dir_path, filename)
            with open(file_path, 'r') as f:
                all_results[epoch] = json.load(f)
    
    # 按轮次编号排序
    sorted_results = dict(sorted(all_results.items()))
    return sorted_results
    
from ray.rllib.algorithms.ppo.ppo_torch_policy import PPOTorchPolicy
import torch

class CustomPPOPolicy(PPOTorchPolicy):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.prev_hidden_states = None
        self.hidden_state_history = []
        self.max_history = 10
        
    def compute_behavior_stability_loss(self, hidden_states, lstm_state):
        """计算LSTM隐状态的稳定性损失"""
        if lstm_state is None:
            return torch.tensor(0.0).to(hidden_states.device)
            
        # 存储当前隐状态
        current_state = lstm_state[0].detach()  # 只使用h_state
        
        # 更新历史
        self.hidden_state_history.append(current_state)
        if len(self.hidden_state_history) > self.max_history:
            self.hidden_state_history.pop(0)
            
        # 如果历史为空，返回0损失
        if len(self.hidden_state_history) < 2:
            return torch.tensor(0.0).to(hidden_states.device)
            
        # 计算隐状态变化的损失
        state_changes = []
        for i in range(len(self.hidden_state_history)-1):
            state_diff = torch.norm(
                self.hidden_state_history[i+1] - self.hidden_state_history[i]
            )
            state_changes.append(state_diff)
            
        stability_loss = torch.mean(torch.stack(state_changes))
        return stability_loss

    def compute_loss(self, train_batch):
        """增加LSTM稳定性的损失计算"""
        # 获取数据
        obs = train_batch["obs"]
        actions = train_batch["actions"]
        advantages = train_batch["advantages"]
        returns = train_batch["value_targets"]
        old_logits = train_batch["action_logits"]
        
        # 1. 获取当前LSTM状态
        lstm_state = None
        if hasattr(self.model, "get_initial_state"):
            lstm_state = self.model.get_initial_state()
        
        # 2. 计算当前策略输出和隐状态
        curr_logits, hidden_states = self.model.forward_policy(
            obs, 
            state=lstm_state,
            seq_lens=train_batch.get("seq_lens")
        )
        
        # 3. 计算行为稳定性损失
        stability_weight = min(1.0, train_batch.get("training_iteration", 0) / 500)
        stability_loss = self.compute_behavior_stability_loss(hidden_states, lstm_state)
        
        # 4. 计算常规PPO损失
        logp_ratio = torch.exp(curr_logits - old_logits)
        policy_loss = -torch.min(
            advantages * logp_ratio,
            advantages * torch.clamp(logp_ratio, 1-self.config["clip_param"], 
                                   1+self.config["clip_param"])
        ).mean()
        
        # 5. 计算值函数损失
        value_pred = self.model.forward_value(obs)
        vf_loss = 0.5 * torch.mean((value_pred - returns) ** 2)
        
        # 6. 计算熵损失(随训练进度降低)
        entropy_loss = -self.config["entropy_coeff"] * \
                      self.model.entropy(curr_logits).mean()
                      
        # 7. 组合所有损失
        total_loss = (
            policy_loss + 
            self.config["vf_loss_coeff"] * vf_loss + 
            entropy_loss +
            stability_weight * stability_loss
        )
        
        return {
            "total_loss": total_loss,
            "policy_loss": policy_loss,
            "vf_loss": vf_loss,
            "entropy": entropy_loss,
            "stability_loss": stability_loss,
            "stability_weight": stability_weight
        }

#保存配置
def save_config(config, file_path):
    """保存配置到JSON文件"""
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    with open(file_path, 'w') as f:
        json.dump(config, f, indent=4)
    print(f"配置已保存至: {file_path}")


# 训练函数 - 适配最新Ray API
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
    defender_lr=2e-4,  # defender使用较大的学习率
    attacker_lr=1e-4,  # attacker使用较小的学习率
    learning_rate_schedule=None,
    history_length=5,
    interval=10, # 保存训练结果的间隔
):
    """使用MAPPO训练防御者和攻击者策略 - 适配最新Ray RLlib版本"""
    print("初始化Ray...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True,num_gpus=1,include_dashboard=False)
    print(f"Ray已初始化: {ray.is_initialized()}")
    print(f"Ray可用资源: {ray.available_resources()}")

    # 注册环境
    env_name = "multistage_signaling_v0"
    register_env(env_name, rllib_env_creator)

    # 创建临时环境实例以获取观察和动作空间
    temp_env = create_rllib_env({
        "n_systems": n_systems,
        "max_steps": max_steps,
        "history_length": history_length,
        "use_tracker": False,
        "visualize_tracker": False
    })
    # obs_spaces = {agent: temp_env.observation_space(agent) for agent in temp_env.possible_agents}
    # act_spaces = {agent: temp_env.action_space(agent) for agent in temp_env.possible_agents}

    obs_spaces = temp_env.observation_space
    act_spaces = temp_env.action_space
    temp_env.close()

    # 使用新的 Ray RLlib API 配置
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
                        "lr": defender_lr,  # defender特定的学习率
                    },
                ),
                "attacker_policy": PolicySpec(
                    observation_space=obs_spaces[ATTACKER],
                    action_space=act_spaces[ATTACKER],
                    config={
                        "agent_id": ATTACKER,
                        "lr": attacker_lr,  # attacker特定的学习率
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
            vf_clip_param=10.0,  # 增大值函数裁剪范围
            vf_loss_coeff=1.0,   # 调整值函数损失权重
            entropy_coeff=0.005,  # 添加熵正则化
            gamma=0.99,         # 保持不变
            lambda_=0.97,       # GAE参数
            num_sgd_iter=10,    # 增加每批次的优化次数
            clip_param=0.2,     # 保持不变
            lr=learning_rate,           # 降低学习率提高稳定性
            train_batch_size=train_batch_size,  # 增大训练批量
        )
        .env_runners(  # ✅ 替换 .rollouts()
            num_env_runners=1,  # 增加采样并行度
            batch_mode="truncate_episodes",  # 改变批处理模式
            rollout_fragment_length=train_batch_size,  # 减小片段长度
            
        )
        .debugging(
            log_level="ERROR", 
            log_sys_usage=True
        )
    )
    config.seed = seed  # ✅ 设置随机种子
    # #保存配置
    # save_config(config.to_dict(), os.path.join(checkpoint_path, "config.json"))

    
    algo = config.build()


    print("MAPPO算法创建成功!")

    # 创建检查点目录
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"检查点将保存至: {checkpoint_path}")

    # 开始训练
    print(f"开始训练，共{stop_iters}轮迭代...")
    start_time = time.time()

    train_results = []
    # 创建训练追踪器
    print("创建训练追踪器...") 
    tracker = TrainingTracker()
    
    result_list = []
    # 训练循环
    for i in range(stop_iters):
        iter_start = time.time()
        result = algo.train()
        #指定轮次保存训练结果
        result_list.append(result)
        # 保存训练结果
        if save_training_results_by_interval(checkpoint_path,result_list,i,interval=interval):
            result_list = []
        # 更新追踪器
        tracker.update(i, result)
        
        iter_time = time.time() - iter_start
        
        # 提取详细训练信息
        print(f"\n===== 迭代 {i+1}/{stop_iters}, 用时: {iter_time:.2f}秒 =====")
        # 每N轮保存一次检查点
        checkpoint_interval = min(50, max(10, stop_iters // 10))
        if (i+1) % checkpoint_interval == 0 or (i+1) == stop_iters:
            checkpoint = algo.save(checkpoint_path)
            print(f"检查点已保存: {i}")

    # 训练完成，保存最终检查点
    final_checkpoint = algo.save(checkpoint_path)
    print(f"训练完成，总用时: {(time.time()-start_time)/60:.2f}分钟")
    print(f"最终检查点: {i}")
   


 
    # 清理
    algo.stop()
    if ray.is_initialized():
        ray.shutdown()

    # 返回结果
    return train_results, final_checkpoint, tracker


import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd

class TrainingVisualizer:
    def __init__(self, figsize=(12, 10)):
        self.figsize = figsize
        
    def plot_training_curves(self, tracker, learning_rate=0.0001):
        """绘制训练曲线
        
        Args:
            tracker: TrainingTracker实例,包含训练数据
        """
        # 创建图表网格
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=self.figsize)
        
        # 绘制奖励曲线
        ax1.plot(tracker.episodes, tracker.defender_rewards, label='Defender', color='blue')
        ax1.plot(tracker.episodes, tracker.attacker_rewards, label='Attacker', color='red')
        
        # 添加标准差区域
        window = 5  # 滑动窗口大小
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
        ax1.set_title(f'Training Progress: Rewardslr=({str(learning_rate)})')
        ax1.legend()
        ax1.grid(True)


        
        # 绘制策略损失曲线 
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
        """保存训练曲线图表
        
        Args:
            tracker: TrainingTracker实例
            save_path: 保存路径
        """
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
        #训练参数
        train_batch_size = 512,
        learning_rate=0.01,
        attacker_lr=1e-4,
        defender_lr=2e-4,  
        learning_rate_schedule=None,
        history_length=20,
        interval=10, # 保存训练结果的间隔
        checkpoint_dir_path="./checkpoint" # 检查点保存路径
        ):
    """训练并保存结果"""
    #创建检查点目录
    checkpoint_dir_path = os.path.join(checkpoint_dir_path)
    os.makedirs(checkpoint_dir_path, exist_ok=True)

    # 保存环境配置
    env_config = {
        "n_systems": n_systems,
        "max_steps": max_steps,
        "history_length": history_length
    }
    with open(os.path.join(checkpoint_dir_path, "env_config.json"), "w") as f:
        json.dump(env_config, f, indent=4)
    print(f"环境配置已保存至: {os.path.join(checkpoint_dir_path, 'env_config.json')}")
    
    # 运行训练
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
        interval=interval, # 保存训练结果的间隔
    )

    #保存tracker为pkl文件
    import pickle
    with open(checkpoint_dir_path+"/training_tracker.pkl", "wb") as f:
        pickle.dump(tracker, f)
 
    #可视化
    visualizer = TrainingVisualizer(figsize=(12, 10))
    # 绘制并显示训练曲线
    visualizer.plot_training_curves(tracker,learning_rate=learning_rate)
    # 保存训练曲线
    visualizer.save_plots(tracker, checkpoint_dir_path+"/training_curves.png")


def load_tracker_from_file(checkpoint_dir_path):
    """从文件加载训练跟踪器"""
    #保存tracker为pkl文件
    import pickle
    with open(checkpoint_dir_path+"/training_tracker.pkl", "rb") as f:
        tracker = pickle.load(f)
    return tracker



TRAINING_ITERATIONS = 400  # 训练迭代次数
USE_LSTM = True
LSTM_CELL_SIZE = 128 # LSTM单元大小
SEED = 52
CHECKPOINT_DIR_PATH = "checkpoints"  # 检查点保存路径

# 环境参数
N_SYSTEMS = 6 # 系统主机数量
MAX_STEPS = 20 # 最大攻防步数
HISTORY_LENGTH = 10 # 最大观测长度记忆
# 训练参数
LEARNING_RATE = 1e-4 # 学习率
#训练batch大小
TRAIN_BATCH_SIZE = 128
# 动态学习率
LEARNING_RATE_SCHEDULE = None

if __name__ == "__main__":
    learn_rate_name = ["lr1", "lr2", "lr3","lr4"]
    learn_rate_list = [2e-4, 1e-4, 7e-5, 5e-5]
    # 训练不同学习率的模型
    attcker_lr_list = [2e-4, 1e-4, 7e-5, 5e-5]
    defender_lr_list = [2e-4, 1e-4, 7e-5, 5e-5]
    for learn_rate, name,index in zip(learn_rate_list, learn_rate_name,range(len(learn_rate_list))):
        print(f"开始训练，学习率: {learn_rate}")
        train_and_save_results(
            stop_iters=TRAINING_ITERATIONS,
            use_lstm=USE_LSTM,
            lstm_cell_size=LSTM_CELL_SIZE,
            seed=SEED,
            n_systems=N_SYSTEMS,
            max_steps=MAX_STEPS,
            #学习率
            learning_rate=learn_rate,
            attacker_lr = learn_rate,
            defender_lr = learn_rate,
            #训练参数
            train_batch_size = TRAIN_BATCH_SIZE,
            learning_rate_schedule=LEARNING_RATE_SCHEDULE,
            history_length=HISTORY_LENGTH,
            interval=10, # 保存训练结果的间隔
            checkpoint_dir_path=f"{CHECKPOINT_DIR_PATH}_{name}"
        )