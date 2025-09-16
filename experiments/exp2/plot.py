import sys
import os
from .env import *
from .train import parallel_env,rllib_env_creator
from .train import create_rllib_env
from .train import N_SYSTEMS, MAX_STEPS

CHECKPOINT_PATH = "./checkpoints_lr1"

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import os
import ray
from ray.rllib.algorithms.ppo import PPO
import torch
from typing import List, Dict, Tuple
from ray.tune.registry import register_env
import json
plt.rcParams['font.family'] = 'Times New Roman'


def get_model_input_size(env):
    """获取环境的观察空间总维度"""
    try:
        obs_space = env.observation_space[DEFENDER]
        total_size = 0
        
        # 打印完整的观察空间信息
        print("\nObservation space structure:")
        for key, space in obs_space.spaces.items():
            print(f"Space {key}:")
            print(f"  Type: {type(space)}")
            print(f"  Shape: {space.shape if hasattr(space, 'shape') else 'No shape'}")
            
            if isinstance(space, Box):
                size = np.prod(space.shape)
                total_size += size
                print(f"  Size: {size}")
            
        print(f"\nTotal observation size: {total_size}")
        return total_size
        
    except Exception as e:
        print(f"Error calculating model input size: {str(e)}")
        return 60  # 默认值，确保程序不会崩溃

class PolicyAnalyzer:
    def __init__(self, checkpoint_path: str, env_config: dict = None,game_steps: int = 20):
        """初始化策略分析器
            params:
            checkpoint_path: 检查点路径
            env_config: 环境配置
            game_steps: 游戏步数
        
        """
        self.checkpoint_path = checkpoint_path
        
        # 1. 确保环境注册
        if not ray.is_initialized():
            ray.init(ignore_reinit_error=True)
            
        env_name = "multistage_signaling_v0"
        register_env(env_name, lambda config: create_rllib_env(config))
        
        # 2. 从检查点加载算法和配置
        print("正在从检查点加载配置...")
        self.algo = PPO.from_checkpoint(checkpoint_path)
        loaded_config = self.algo.get_config()
        
        # 3. 获取环境配置
        self.env_config = loaded_config.get("env_config", {})
        if env_config:  # 如果外部传入了配置，则更新
            self.env_config.update(env_config)
        #更新环境配置
        self.env_config["max_steps"] = game_steps    
        print(f"加载的环境配置: {self.env_config}")
        
        # 4. 创建环境并获取观察空间维度
        temp_env = create_rllib_env(self.env_config)
        actual_size = get_model_input_size(temp_env)
        print(f"Environment observation size: {actual_size}")
        
        # 5. 获取策略和模型
        defender_policy = self.algo.get_policy("defender_policy")
        model = defender_policy.model
        
        # 6. 尝试获取模型输入维度
        expected_size = actual_size
        try:
            if hasattr(model, 'obs_space') and model.obs_space is not None:
                expected_size = model.obs_space.shape[-1] if hasattr(model.obs_space, 'shape') else None
            if expected_size is None:
                expected_size = actual_size
                print("Warning: Could not get model input size, using environment size instead")
        except Exception as e:
            print(f"Error getting model input size: {str(e)}")
            expected_size = actual_size
        print(f"Model input size: {expected_size}")    
        print(f"Model expected input size: {expected_size}")
        
        # 7. 创建最终环境实例
        self.env = create_rllib_env(self.env_config)
        
    def analyze_defender_policy(self, n_episodes: int = 100) -> Dict:
        signal_probs = {
            "real": {0: [], 1: []},
            "honeypot": {0: [], 1: []}
        }
        
        # 初始化双方策略状态
        defender_policy = self.algo.get_policy("defender_policy")
        attacker_policy = self.algo.get_policy("attacker_policy")
        defender_state = defender_policy.get_initial_state()
        attacker_state = attacker_policy.get_initial_state()
        
        for _ in range(n_episodes):
            obs = self.env.reset()[0]
            done = False
            
            while not done:
                defender_obs = {
                    "current_step": np.expand_dims(obs[DEFENDER]["current_step"].astype(np.float32), axis=0),
                    "system_types": np.expand_dims(obs[DEFENDER]["system_types"].astype(np.int8), axis=0),
                    "last_attacker_actions": np.expand_dims(obs[DEFENDER]["last_attacker_actions"].astype(np.int8), axis=0),
                    "attacker_action_history": np.expand_dims(obs[DEFENDER]["attacker_action_history"].astype(np.int8), axis=0),
                    "defender_signal_history": np.expand_dims(obs[DEFENDER]["defender_signal_history"].astype(np.int8), axis=0)
                }
                
                for key in defender_obs:
                    if isinstance(defender_obs[key], np.ndarray):
                        defender_obs[key] = defender_obs[key].astype(np.float32)
                
                defender_action, defender_state, info = defender_policy.compute_single_action(
                    defender_obs,
                    state=defender_state,
                    explore=False,
                    full_fetch=True
                )
                defender_action = np.array(defender_action, dtype=np.int8)

                if "action_dist_inputs" in info:
                    action_probs = torch.sigmoid(torch.tensor(info["action_dist_inputs"])).numpy()
                    
                    system_types = defender_obs["system_types"][0]
                    for i, sys_type in enumerate(system_types):
                        if sys_type == THETA_REAL:
                            signal_probs["real"][0].append(1 - action_probs[i])
                            signal_probs["real"][1].append(action_probs[i])
                        else:
                            signal_probs["honeypot"][0].append(1 - action_probs[i])
                            signal_probs["honeypot"][1].append(action_probs[i])

                attacker_input = {
                    "current_step": obs[ATTACKER]["current_step"],
                    "current_defender_signals": obs[ATTACKER]["current_defender_signals"],
                    "attacker_action_history": obs[ATTACKER]["attacker_action_history"],
                    "defender_signal_history": obs[ATTACKER]["defender_signal_history"],
                    "belief_real": obs[ATTACKER]["belief_real"]
                }
                
                attacker_action, attacker_state, _ = attacker_policy.compute_single_action(
                    attacker_input,
                    state=attacker_state,
                    explore=False
                )
                attacker_action = np.array(attacker_action, dtype=np.int8)

                action_dict = {
                    DEFENDER: defender_action,
                    ATTACKER: attacker_action
                }
                
                step_result = self.env.step(action_dict)
                obs = step_result[0]
                terminations = step_result[2]
                truncations = step_result[3]
                done = all(terminations.values()) or all(truncations.values())

        result = {
            "real": np.array([signal_probs["real"][0], signal_probs["real"][1]]),
            "honeypot": np.array([signal_probs["honeypot"][0], signal_probs["honeypot"][1]])
        }
        
        return result
                
    def plot_defender_strategy(self, signal_probs: Dict, save_path: str = "./figures", font_size: int = 18):
        """绘制防御者策略分布
        
        Args:
            signal_probs: 包含不同系统类型下不同信号的概率的字典
        """
        # 绘制真实系统下的信号分布（图1）
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.size': font_size})
        
        # 准备数据
        real_data = [
            signal_probs["real"][0],  # 真实系统发送sN信号的概率
            signal_probs["real"][1]   # 真实系统发送sH信号的概率
        ]
        
        labels = ['Normal Signal', 'Honeypot Signal']
            
        # 对violinplot的调用进行修改
        violin_parts = plt.violinplot(real_data, showmedians=True)
        # 将中位数线改为红色
        violin_parts['cmedians'].set_color('green')
        violin_parts['cmedians'].set_linewidth(2)
        violin_parts['cmedians'].set_label("Median")
             
        plt.boxplot(real_data, showfliers=False)
        plt.xticks([1, 2], labels, fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel('Signal Selection Probability', fontsize=font_size)
        # plt.title('Defender Strategy on Real Systems (θ1)', fontsize=font_size)
        plt.grid(True, linestyle="--", alpha=0.4)
        
        # 添加均值标记
        real_means = [np.mean(probs) if len(probs) > 0 else 0 for probs in real_data]
        plt.plot([1, 2], real_means, 'r*', label='Mean')
        plt.legend(fontsize=font_size)
        
        plt.tight_layout()
        # 保存图片
        plt.savefig(save_path+"/defender_real_system_strategy.png", dpi=300)
        plt.savefig(save_path+"/defender_real_system_strategy.pdf")
        plt.close()
        
        # 绘制蜜罐系统下的信号分布（图2）
        plt.figure(figsize=(10, 8))
        
        # 准备数据
        honeypot_data = [
            signal_probs["honeypot"][0],  # 蜜罐系统发送sN信号的概率
            signal_probs["honeypot"][1]   # 蜜罐系统发送sH信号的概率
        ]
        

        # 对violinplot的调用进行修改
        violin_parts = plt.violinplot(honeypot_data, showmedians=True)
        # 将中位数线改为红色
        violin_parts['cmedians'].set_color('green')
        violin_parts['cmedians'].set_linewidth(2)
        violin_parts['cmedians'].set_label("Median")


        plt.boxplot(honeypot_data, showfliers=False)
        plt.xticks([1, 2], labels)
        plt.ylabel('Signal Selection Probability', fontsize=font_size)
        # plt.title('Defender Strategy on Honeypot Systems (θ2)', fontsize=font_size)
        plt.grid(True, linestyle="--", alpha=0.4)
        
        # 添加均值标记
        honeypot_means = [np.mean(probs) if len(probs) > 0 else 0 for probs in honeypot_data]
        plt.plot([1, 2], honeypot_means, 'r*', label='Mean')
        plt.legend(fontsize=font_size)
        
        plt.tight_layout()
        # 保存图片
        plt.savefig(save_path+"/defender_honeypot_system_strategy.png", dpi=300)
        plt.savefig(save_path+"/defender_honeypot_system_strategy.pdf")
        plt.close()

    def analyze_attacker_policy(self, n_episodes: int = 100) -> Dict:
        """分析攻击者策略"""
        attack_probs = {
            "real": {
                "sN": [],  # 真实系统下收到sN信号时的攻击概率
                "sH": []   # 真实系统下收到sH信号时的攻击概率
            },
            "honeypot": {
                "sN": [],  # 蜜罐系统下收到sN信号时的攻击概率
                "sH": []   # 蜜罐系统下收到sH信号时的攻击概率
            }
        }
        
        # 初始化双方策略状态
        defender_policy = self.algo.get_policy("defender_policy")
        attacker_policy = self.algo.get_policy("attacker_policy")
        defender_state = defender_policy.get_initial_state()
        attacker_state = attacker_policy.get_initial_state()
        
        for _ in range(n_episodes):
            obs = self.env.reset()[0]  # 只取第一个返回值(observations)
            done = False
            
            while not done:
                # 1. 获取防御者动作
                defender_obs = {
                    "current_step": np.expand_dims(obs[DEFENDER]["current_step"].astype(np.float32), axis=0),
                    "system_types": np.expand_dims(obs[DEFENDER]["system_types"].astype(np.int8), axis=0),
                    "last_attacker_actions": np.expand_dims(obs[DEFENDER]["last_attacker_actions"].astype(np.int8), axis=0),
                    "attacker_action_history": np.expand_dims(obs[DEFENDER]["attacker_action_history"].astype(np.int8), axis=0),
                    "defender_signal_history": np.expand_dims(obs[DEFENDER]["defender_signal_history"].astype(np.int8), axis=0)
                }
                
                defender_action, defender_state, _ = defender_policy.compute_single_action(
                    defender_obs,
                    state=defender_state,
                    explore=False
                )
                defender_action = np.array(defender_action, dtype=np.int8)
                
                # 2. 获取攻击者动作分布
                attacker_obs = {
                    "current_step": np.expand_dims(obs[ATTACKER]["current_step"].astype(np.float32), axis=0),
                    "current_defender_signals": np.expand_dims(obs[ATTACKER]["current_defender_signals"].astype(np.int8), axis=0),
                    "attacker_action_history": np.expand_dims(obs[ATTACKER]["attacker_action_history"].astype(np.int8), axis=0),
                    "defender_signal_history": np.expand_dims(obs[ATTACKER]["defender_signal_history"].astype(np.int8), axis=0),
                    "belief_real": np.expand_dims(obs[ATTACKER]["belief_real"].astype(np.float32), axis=0)
                }
                
                attacker_action, attacker_state, info = attacker_policy.compute_single_action(
                    attacker_obs,
                    state=attacker_state,
                    explore=False,
                    full_fetch=True  # 获取完整信息包括动作分布
                )
                attacker_action = np.array(attacker_action, dtype=np.int8)
                
                # 3. 记录攻击概率，区分系统类型
                if "action_dist_inputs" in info:
                    action_dist = info["action_dist_inputs"]
                    signals = attacker_obs["current_defender_signals"][0]
                    system_types = defender_obs["system_types"][0]
                    
                    for i, (signal, system_type) in enumerate(zip(signals, system_types)):
                        prob = torch.sigmoid(torch.tensor(action_dist[i])).item()
                        
                        if system_type == 0:  # 真实系统
                            if signal == 0:  # sN
                                attack_probs["real"]["sN"].append(prob)
                            else:  # sH
                                attack_probs["real"]["sH"].append(prob)
                        else:  # 蜜罐系统
                            if signal == 0:  # sN
                                attack_probs["honeypot"]["sN"].append(prob)
                            else:  # sH
                                attack_probs["honeypot"]["sH"].append(prob)
                
                # 4. 构建联合动作字典并执行环境步进
                action_dict = {
                    DEFENDER: defender_action,
                    ATTACKER: attacker_action
                }
                
                step_result = self.env.step(action_dict)
                obs = step_result[0]
                terminations = step_result[2]
                truncations = step_result[3]
                done = all(terminations.values()) or all(truncations.values())

        return attack_probs
    
    
    def plot_attacker_strategy(self, attack_probs: Dict,save_path: str = "./figures",font_size: int = 18):
        """绘制攻击者策略分布
        
        Args:
            attack_probs: 包含不同系统类型下不同信号的攻击概率的字典
        """
        # 绘制真实系统下的攻击概率分布（图1）
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.size': font_size})
        # 准备数据
        real_data = [
            attack_probs["real"]["sN"],  # 真实系统下收到sN信号时的攻击概率
            attack_probs["real"]["sH"]   # 真实系统下收到sH信号时的攻击概率
        ]
        
        labels = ['Normal Signal', 'Honeypot Signal']
        
        # 对violinplot的调用进行修改
        violin_parts = plt.violinplot(real_data, showmedians=True)
        # 将中位数线改为红色
        violin_parts['cmedians'].set_color('green')
        violin_parts['cmedians'].set_linewidth(2)
        violin_parts['cmedians'].set_label("Median")
        plt.boxplot(real_data, showfliers=False)
        plt.xticks([1, 2], labels,fontsize = font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel('Attack Probability',fontsize=font_size)
        # plt.title('Attacker Strategy on Real Systems (θ1)',fontsize=font_size)
        plt.grid(True, linestyle="--", alpha=0.4)
        
        # 添加均值标记
        real_means = [np.mean(probs) if probs else 0 for probs in real_data]
        plt.plot([1, 2], real_means, 'r*', label='Mean')
        plt.legend(fontsize=font_size)
        
        plt.tight_layout()
        # 保存图片
        plt.savefig(save_path+"/attacker_real_system_strategy.png", dpi=300)
        plt.savefig(save_path+"/attacker_real_system_strategy.pdf")
        
        # 绘制蜜罐系统下的攻击概率分布（图2）
        plt.figure(figsize=(10, 8))
        
        # 准备数据
        honeypot_data = [
            attack_probs["honeypot"]["sN"],  # 蜜罐系统下收到sN信号时的攻击概率
            attack_probs["honeypot"]["sH"]   # 蜜罐系统下收到sH信号时的攻击概率
        ]
        
        # 对violinplot的调用进行修改
        violin_parts = plt.violinplot(honeypot_data, showmedians=True)
        # 将中位数线改为红色
        violin_parts['cmedians'].set_color('green')
        violin_parts['cmedians'].set_linewidth(2)
        violin_parts['cmedians'].set_label("Median")
        plt.boxplot(honeypot_data, showfliers=False)
        plt.xticks([1, 2], labels,fontsize=font_size)
        plt.yticks(fontsize=font_size)
        plt.ylabel('Attack Probability',fontsize=font_size)
        # plt.title('Attacker Strategy on Honeypot Systems (θ2)',fontsize=font_size)
        plt.grid(True, linestyle="--", alpha=0.4)
        
        # 添加均值标记
        honeypot_means = [np.mean(probs) if probs else 0 for probs in honeypot_data]
        plt.plot([1, 2], honeypot_means, 'r*', label='Mean')
        plt.legend(fontsize=font_size)
        
        plt.tight_layout()
        # 保存图片
        plt.savefig(save_path+"/attacker_honeypot_system_strategy.png", dpi=300)
        plt.savefig(save_path+"/attacker_honeypot_system_strategy.pdf")

    def plot_belief_trajectory(self, beliefs: List[float],save_path: str = "./figures",font_size: int = 18):
        """绘制信念演化轨迹"""
        plt.figure(figsize=(10, 8))
        plt.rcParams.update({'font.size': font_size})
        plt.plot(beliefs, marker='o')
        plt.xlabel('Time Step',fontsize=font_size)
        plt.ylabel('Attack Belief',fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        # plt.title('Belief Evolution in Deception Scenario',fontsize=font_size)
        # plt.grid(True)
        plt.tight_layout()
        plt.savefig(save_path+"/attacker_belief_trajectory.png", dpi=300)
        plt.savefig(save_path+"/attacker_belief_trajectory.pdf")

    def analyze_belief_evolution(self, n_episodes: int = 10) -> List[Dict]:
        """分析攻击者信念的演化过程
        
        Args:
            n_episodes: 要分析的场景数量
            
        Returns:
            belief_trajectories: 包含多个场景的信念演化轨迹列表
        """
        belief_trajectories = []
        
        defender_policy = self.algo.get_policy("defender_policy")
        attacker_policy = self.algo.get_policy("attacker_policy")
        
        for episode in range(n_episodes):
            # 记录当前场景的信念演化
            episode_beliefs = []
            obs = self.env.reset()[0]
            done = False
            defender_state = defender_policy.get_initial_state()
            attacker_state = attacker_policy.get_initial_state()
            
            while not done:
                # 获取防御者动作
                defender_obs = {
                    "current_step": np.expand_dims(obs[DEFENDER]["current_step"].astype(np.float32), axis=0),
                    "system_types": np.expand_dims(obs[DEFENDER]["system_types"].astype(np.int8), axis=0),
                    "last_attacker_actions": np.expand_dims(obs[DEFENDER]["last_attacker_actions"].astype(np.int8), axis=0),
                    "attacker_action_history": np.expand_dims(obs[DEFENDER]["attacker_action_history"].astype(np.int8), axis=0),
                    "defender_signal_history": np.expand_dims(obs[DEFENDER]["defender_signal_history"].astype(np.int8), axis=0)
                }
                
                defender_action, defender_state, _ = defender_policy.compute_single_action(
                    defender_obs,
                    state=defender_state,
                    explore=False
                )
                defender_action = np.array(defender_action, dtype=np.int8)
                
                # 获取攻击者行为分布
                attacker_obs = {
                    "current_step": np.expand_dims(obs[ATTACKER]["current_step"].astype(np.float32), axis=0),
                    "current_defender_signals": np.expand_dims(obs[ATTACKER]["current_defender_signals"].astype(np.int8), axis=0),
                    "attacker_action_history": np.expand_dims(obs[ATTACKER]["attacker_action_history"].astype(np.int8), axis=0),
                    "defender_signal_history": np.expand_dims(obs[ATTACKER]["defender_signal_history"].astype(np.int8), axis=0),
                    "belief_real": np.expand_dims(obs[ATTACKER]["belief_real"].astype(np.float32), axis=0)
                }
                
                # 获取完整的动作分布信息
                attacker_action, attacker_state, info = attacker_policy.compute_single_action(
                    attacker_obs,
                    state=attacker_state,
                    explore=False,
                    full_fetch=True
                )
                attacker_action = np.array(attacker_action, dtype=np.int8)
                
                # 计算当前时间步的信念状态
                if "action_dist_inputs" in info:
                    action_dist = info["action_dist_inputs"]
                    # 使用sigmoid将logits转换为概率
                    attack_probs = torch.sigmoid(torch.tensor(action_dist)).numpy()
                    # 计算平均信念（攻击概率可以视为系统是真实的信念）
                    mean_belief = np.mean(attack_probs)
                    episode_beliefs.append(mean_belief)
                
                # 执行环境步进
                action_dict = {
                    DEFENDER: defender_action,
                    ATTACKER: attacker_action
                }
                step_result = self.env.step(action_dict)
                obs = step_result[0]
                terminations = step_result[2]
                truncations = step_result[3]
                done = all(terminations.values()) or all(truncations.values())
            
            belief_trajectories.append(episode_beliefs)
        
        return belief_trajectories

    def plot_belief_trajectories(self, belief_trajectories: List[List[float]], save_path: str = "./figures",font_size: int = 18):
        """绘制多个场景的信念演化轨迹
        
        Args:
            belief_trajectories: 包含多个场景信念演化数据的列表
            output_dir: 可选的输出目录路径
        """
        plt.figure(figsize=(12, 8))
        plt.rcParams.update({'font.size': font_size})
        # 绘制所有轨迹
        for i, beliefs in enumerate(belief_trajectories):
            plt.plot(beliefs, alpha=0.3, color='blue', label='Individual Trajectory' if i == 0 else None)
        
        # 计算并绘制平均轨迹
        max_length = max(len(traj) for traj in belief_trajectories)
        aligned_trajectories = []
        for traj in belief_trajectories:
            # 将较短的轨迹通过复制最后一个值来延长到最大长度
            padded_traj = traj + [traj[-1]] * (max_length - len(traj))
            aligned_trajectories.append(padded_traj)
        
        mean_trajectory = np.mean(aligned_trajectories, axis=0)
        std_trajectory = np.std(aligned_trajectories, axis=0)
        
        time_steps = range(max_length)
        plt.plot(time_steps, mean_trajectory, color='red', linewidth=2, label='Mean Trajectory')
        plt.fill_between(time_steps, 
                        mean_trajectory - std_trajectory,
                        mean_trajectory + std_trajectory,
                        color='red', alpha=0.2, label='±1 std')
        
        plt.xlabel('Time Steps',fontsize=font_size)
        plt.ylabel('Attack Belief',fontsize=font_size)
        plt.xticks(fontsize=font_size)
        plt.yticks(fontsize=font_size)
        # plt.title('Evolution of Attacker Beliefs Over Time',fontsize=font_size)
        # plt.grid(True, alpha=0.3)
        plt.legend(fontsize=font_size)
        plt.tight_layout()
        # 保存图片
        plt.savefig(save_path+"/belief_evolution.png", dpi=300)
        plt.savefig(save_path+"/belief_evolution.pdf")
        plt.close()
            
    def run_full_analysis(self, save_path: str = "./figures",step: int = 20):
        """运行完整的策略分析并保存结果
          param save_path: 保存结果的路径
          param step: 分析的步数
          param game_step: 游戏步数
        """
        os.makedirs(save_path, exist_ok=True)
        
        try:
            # 1. 分析防御者策略
            print("分析防御者策略...")
            defender_probs = self.analyze_defender_policy(n_episodes=step)
            self.plot_defender_strategy(defender_probs,save_path=save_path,font_size=25)
            #保存数据json(中位数和均值)
            defender_data = {
                "real": {
                    "sN": {
                        "mean": float(np.mean(defender_probs["real"][0])),
                        "median": float(np.median(defender_probs["real"][0]))
                    },
                    "sH": {
                        "mean": float(np.mean(defender_probs["real"][1])),
                        "median": float(np.median(defender_probs["real"][1]))
                    }
                },
                "honeypot": {
                    "sN": {
                        "mean": float(np.mean(defender_probs["honeypot"][0])),
                        "median": float(np.median(defender_probs["honeypot"][0]))
                    },
                    "sH": {
                        "mean": float(np.mean(defender_probs["honeypot"][1])),
                        "median": float(np.median(defender_probs["honeypot"][1]))
                    }
                }
            }
            #计算均值差
            defender_data["real"]["mean_diff"] = abs(defender_data["real"]["sN"]["mean"] - defender_data["real"]["sH"]["mean"])
            defender_data["honeypot"]["mean_diff"] = abs(defender_data["honeypot"]["sN"]["mean"] - defender_data["honeypot"]["sH"]["mean"])
            #保存数据json
            with open(os.path.join(save_path, "defender_data.json"), 'w') as f:
                json.dump(defender_data, f,indent=4)
            
            # 2. 分析攻击者策略
            print("分析攻击者策略...")
            attacker_probs = self.analyze_attacker_policy(n_episodes=step)
            self.plot_attacker_strategy(attacker_probs,save_path=save_path,font_size=25)
            #保存数据json(中位数和均值)
            attacker_data = {
                "real": {
                    "sN": {
                        "mean": float(np.mean(attacker_probs["real"]["sN"])),
                        "median": float(np.median(attacker_probs["real"]["sN"]))
                    },
                    "sH": {
                        "mean": float(np.mean(attacker_probs["real"]["sH"])),
                        "median": float(np.median(attacker_probs["real"]["sH"]))
                    }
                },
                "honeypot": {
                    "sN": {
                        "mean": float(np.mean(attacker_probs["honeypot"]["sN"])),
                        "median": float(np.median(attacker_probs["honeypot"]["sN"]))
                    },
                    "sH": {
                        "mean": float(np.mean(attacker_probs["honeypot"]["sH"])),
                        "median": float(np.median(attacker_probs["honeypot"]["sH"]))
                    }
                }
            }
            #计算均值差
            attacker_data["real"]["mean_diff"] = abs(attacker_data["real"]["sN"]["mean"] - attacker_data["real"]["sH"]["mean"])
            attacker_data["honeypot"]["mean_diff"] = abs(attacker_data["honeypot"]["sN"]["mean"] - attacker_data["honeypot"]["sH"]["mean"])

            with open(os.path.join(save_path, "attacker_data.json"), 'w') as f:
                json.dump(attacker_data, f,indent=4)
            
           
            
            # 3. 分析信念演化
            print("分析信念演化...")
            belief_trajectories = self.analyze_belief_evolution(n_episodes=step)
            self.plot_belief_trajectories(belief_trajectories, save_path,font_size=25)
            
            # 保存信念轨迹数据
            # 保存数值结果
            np.save(os.path.join(save_path, "belief_trajectories.npy"), belief_trajectories)
            np.save(os.path.join(save_path, "defender_probs.npy"), defender_probs)
            np.save(os.path.join(save_path, "attacker_probs.npy"), attacker_probs)


            
        except Exception as e:
            print(f"分析过程中出现错误: {str(e)}")
            raise
        finally:
            plt.close('all')
        
    def __del__(self):
        """清理资源"""
        if ray.is_initialized():
            ray.shutdown()


# 找到最新的检查点
def get_latest_checkpoint(checkpoint_dir: str) -> str:
    """获取最新的检查点路径"""
    import glob
    import os
    
    # 确保使用绝对路径
    checkpoint_dir = os.path.abspath(checkpoint_dir)
    # 获取所有检查点文件
    if not checkpoint_dir:
        raise ValueError(f"在 {checkpoint_dir} 中未找到检查点文件")
    return checkpoint_dir

def load_env_config(checkpoint_dir: str) -> dict:
    """从检查点目录加载环境配置"""
    config_path = os.path.join(checkpoint_dir, "env_config.json")
    if not os.path.exists(config_path):
        print(f"警告：未找到环境配置文件 {config_path}，将使用默认配置")
        return {
            "n_systems": N_SYSTEMS,
            "max_steps": MAX_STEPS,
            "history_length": MAX_STEPS
        }
    
    with open(config_path, 'r') as f:
        return json.load(f)

def run_analysis(checkpoint_path: str,game_steps: int):
    """运行分析"""
    # 获取最新检查点路径
    latest_checkpoint = get_latest_checkpoint(checkpoint_path)
    print(f"使用检查点: {latest_checkpoint}")

    # 创建分析器实例
    analyzer = PolicyAnalyzer(checkpoint_path=latest_checkpoint,game_steps=game_steps)

    # 运行完整分析
    save_path = "./figures/game_steps_"+str(game_steps)
    os.makedirs(save_path, exist_ok=True)
    analyzer.run_full_analysis(save_path=save_path)
        
    # 确保清理Ray资源
    if ray.is_initialized():
        ray.shutdown()

if __name__ == "__main__":
    lr_list = ["lr1", "lr2", "lr3", "lr4"]
    game_step_list = [15,30,60,120]
    for lr in ["lr2"]:
        checkpoint_path = f"./checkpoints_{lr}"
        print(f"\n分析学习率配置: {lr}")
        for game_steps in game_step_list:
            run_analysis(checkpoint_path,game_steps)