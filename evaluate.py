import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import pickle
import ray
from ray.rllib.algorithms.ppo import PPO
from env import parallel_env, DEFENDER, ATTACKER, MAX_STEPS, N_SYSTEMS
from train import create_rllib_env, policy_mapping_fn

# 设置 matplotlib 样式
plt.style.use('seaborn-v0_8-whitegrid')
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

def load_algorithm(checkpoint_path, config=None):
    """加载训练好的算法模型"""
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    
    # 如果提供了配置，使用它，否则使用默认配置
    if config is None:
        config = {
            "framework": "torch",
            "disable_env_checking": True,
            "model": {
                "use_lstm": True,
                "lstm_cell_size": 128,
                "fcnet_hiddens": [128, 128],
                "centralized_critic": True
            }
        }
    
    # 创建算法实例并加载检查点
    algo = PPO(config=config)
    algo.restore(checkpoint_path)
    
    return algo

def evaluate_policy(algo, num_episodes=100, render=False):
    """评估策略表现"""
    # 创建环境
    env = parallel_env()
    
    defender_rewards = []
    attacker_rewards = []
    episode_lengths = []
    honeypot_signals = []  # 记录防御者使用的蜜罐信号
    attack_actions = []    # 记录攻击者采取的攻击行动
    
    # 历史信息分析数据
    history_analysis = {
        "defender": {
            "signal_history_impact": [],
            "action_history_impact": []
        },
        "attacker": {
            "signal_history_impact": [],
            "action_history_impact": []
        }
    }
    
    # 对系统类型进行分析的数据
    real_system_data = []  # [(信号, 行动, 奖励), ...]
    honeypot_system_data = []  # [(信号, 行动, 奖励), ...]
    
    print(f"运行 {num_episodes} 个评估回合...")
    
    for ep in range(num_episodes):
        if ep % 10 == 0:
            print(f"回合 {ep}/{num_episodes}")
            
        observations = env.reset()
        ep_defender_reward = 0
        ep_attacker_reward = 0
        ep_steps = 0
        ep_signals = []
        ep_actions = []
        
        # 记录系统类型
        system_types = env.env.system_types
        
        done = {"__all__": False}
        while not done["__all__"]:
            # 获取策略动作
            actions = {}
            for agent, obs in observations.items():
                policy_id = policy_mapping_fn(agent)
                # 处理Dict类型的观察空间
                if isinstance(obs, dict):
                    # 提取关键观察信息
                    if agent == DEFENDER:
                        current_obs = {
                            "current_step": obs["current_step"],
                            "system_types": obs["system_types"],
                            "attacker_actions": obs["attacker_actions"],
                            "attacker_action_history": obs["attacker_action_history"],
                            "defender_signal_history": obs["defender_signal_history"]
                        }
                    else:  # ATTACKER
                        current_obs = {
                            "current_step": obs["current_step"],
                            "defender_signals": obs["defender_signals"],
                            "attacker_action_history": obs["attacker_action_history"],
                            "defender_signal_history": obs["defender_signal_history"]
                        }
                    
                    # 分析历史信息的影响
                    if agent == DEFENDER:
                        history_analysis["defender"]["signal_history_impact"].append(
                            np.mean(obs["defender_signal_history"])
                        )
                        history_analysis["defender"]["action_history_impact"].append(
                            np.mean(obs["attacker_action_history"])
                        )
                    else:
                        history_analysis["attacker"]["signal_history_impact"].append(
                            np.mean(obs["defender_signal_history"])
                        )
                        history_analysis["attacker"]["action_history_impact"].append(
                            np.mean(obs["attacker_action_history"])
                        )
                    
                    action = algo.compute_single_action(current_obs, policy_id=policy_id)
                else:
                    action = algo.compute_single_action(obs, policy_id=policy_id)
                actions[agent] = action
            
            # 记录每步的行动
            if DEFENDER in actions and ATTACKER in actions:
                ep_signals.append(actions[DEFENDER])
                ep_actions.append(actions[ATTACKER])
            
            # 执行动作
            observations, rewards, done, _, info = env.step(actions)
            
            # 累计奖励
            if DEFENDER in rewards:
                ep_defender_reward += rewards[DEFENDER]
            if ATTACKER in rewards:
                ep_attacker_reward += rewards[ATTACKER]
                
            # 收集按系统类型分类的数据
            if ep_steps > 0:  # 从第二步开始，防御者和攻击者都行动过
                for i in range(N_SYSTEMS):
                    if system_types[i] == 0:  # 真实系统
                        real_system_data.append((
                            ep_signals[-2][i],  # 防御者信号
                            ep_actions[-1][i],  # 攻击者行动
                            rewards.get(DEFENDER, 0),  # 防御者奖励
                            rewards.get(ATTACKER, 0)   # 攻击者奖励
                        ))
                    else:  # 蜜罐系统
                        honeypot_system_data.append((
                            ep_signals[-2][i],  # 防御者信号
                            ep_actions[-1][i],  # 攻击者行动
                            rewards.get(DEFENDER, 0),  # 防御者奖励
                            rewards.get(ATTACKER, 0)   # 攻击者奖励
                        ))
            
            if render:
                env.render()
                
            ep_steps += 1
        
        # 记录回合数据
        defender_rewards.append(ep_defender_reward)
        attacker_rewards.append(ep_attacker_reward)
        episode_lengths.append(ep_steps)
        honeypot_signals.append(np.mean([s.count(1) for s in ep_signals]) / N_SYSTEMS if ep_signals else 0)
        attack_actions.append(np.mean([a.count(1) for a in ep_actions]) / N_SYSTEMS if ep_actions else 0)
    
    # 计算统计数据
    results = {
        "defender_rewards": {
            "mean": np.mean(defender_rewards),
            "std": np.std(defender_rewards),
            "min": np.min(defender_rewards),
            "max": np.max(defender_rewards),
            "raw": defender_rewards
        },
        "attacker_rewards": {
            "mean": np.mean(attacker_rewards),
            "std": np.std(attacker_rewards),
            "min": np.min(attacker_rewards),
            "max": np.max(attacker_rewards),
            "raw": attacker_rewards
        },
        "episode_lengths": {
            "mean": np.mean(episode_lengths),
            "std": np.std(episode_lengths),
            "raw": episode_lengths
        },
        "honeypot_signals": {
            "mean": np.mean(honeypot_signals),
            "std": np.std(honeypot_signals),
            "raw": honeypot_signals
        },
        "attack_actions": {
            "mean": np.mean(attack_actions),
            "std": np.std(attack_actions),
            "raw": attack_actions
        },
        "real_system_data": real_system_data,
        "honeypot_system_data": honeypot_system_data,
        "history_analysis": history_analysis
    }
    
    return results

def plot_rewards_over_time(train_results, save_path=None):
    """绘制训练过程中奖励变化"""
    df = pd.DataFrame([
        {
            "iteration": i,
            "defender_reward": r.get("policy_reward_mean", {}).get("defender_policy", float('nan')),
            "attacker_reward": r.get("policy_reward_mean", {}).get("attacker_policy", float('nan')),
        }
        for i, r in enumerate(train_results)
    ])
    
    # 计算移动平均
    window = 10
    df['defender_ma'] = df['defender_reward'].rolling(window=window).mean()
    df['attacker_ma'] = df['attacker_reward'].rolling(window=window).mean()
    
    # 绘制图表
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 绘制原始数据（点）
    ax.scatter(df['iteration'], df['defender_reward'], alpha=0.3, color='blue', label='_nolegend_')
    ax.scatter(df['iteration'], df['attacker_reward'], alpha=0.3, color='red', label='_nolegend_')
    
    # 绘制移动平均（线）
    ax.plot(df['iteration'], df['defender_ma'], color='blue', linewidth=2, label='Defender Reward (MA)')
    ax.plot(df['iteration'], df['attacker_ma'], color='red', linewidth=2, label='Attacker Reward (MA)')
    
    # 添加标签和图例
    ax.set_xlabel('Training Iteration')
    ax.set_ylabel('Average Episode Reward')
    ax.set_title('Agent Rewards During Training')
    ax.legend()
    ax.grid(True)
    
    # 保存图表
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()

def plot_evaluation_results(eval_results, save_dir=None):
    """绘制评估结果图表"""
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 1. 智能体奖励分布直方图
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # 防御者奖励分布
    ax1.hist(eval_results["defender_rewards"]["raw"], bins=20, alpha=0.7, color='blue')
    ax1.axvline(eval_results["defender_rewards"]["mean"], color='k', linestyle='dashed', linewidth=2)
    ax1.set_xlabel('Reward')
    ax1.set_ylabel('Frequency')
    ax1.set_title(f'Defender Rewards\nMean: {eval_results["defender_rewards"]["mean"]:.2f} ± {eval_results["defender_rewards"]["std"]:.2f}')
    
    # 攻击者奖励分布
    ax2.hist(eval_results["attacker_rewards"]["raw"], bins=20, alpha=0.7, color='red')
    ax2.axvline(eval_results["attacker_rewards"]["mean"], color='k', linestyle='dashed', linewidth=2)
    ax2.set_xlabel('Reward')
    ax2.set_title(f'Attacker Rewards\nMean: {eval_results["attacker_rewards"]["mean"]:.2f} ± {eval_results["attacker_rewards"]["std"]:.2f}')
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'reward_distributions.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 2. 蜜罐信号与攻击行动比例
    fig, ax = plt.subplots(figsize=(10, 6))
    
    labels = ['Honeypot Signal Rate', 'Attack Action Rate']
    means = [eval_results["honeypot_signals"]["mean"], eval_results["attack_actions"]["mean"]]
    stds = [eval_results["honeypot_signals"]["std"], eval_results["attack_actions"]["std"]]
    
    x = np.arange(len(labels))
    width = 0.35
    
    ax.bar(x, means, width, yerr=stds, capsize=10, color=['blue', 'red'])
    ax.set_ylabel('Rate')
    ax.set_title('Honeypot Signaling and Attack Rates')
    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylim(0, 1)
    
    for i, v in enumerate(means):
        ax.text(i, v + 0.05, f'{v:.2f}', ha='center')
    
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'action_rates.png'), dpi=300, bbox_inches='tight')
    plt.show()
    
    # 3. 策略分析：系统类型与行动
    # 将数据转换为更易于分析的格式
    real_data = np.array(eval_results["real_system_data"])
    honeypot_data = np.array(eval_results["honeypot_system_data"])
    
    if len(real_data) > 0 and len(honeypot_data) > 0:
        # 准备数据：按系统类型、防御者信号和攻击者行动进行分组
        categories = [
            ("Real", "Normal", "Attack"),
            ("Real", "Normal", "Retreat"),
            ("Real", "Honeypot", "Attack"),
            ("Real", "Honeypot", "Retreat"),
            ("Honeypot", "Normal", "Attack"),
            ("Honeypot", "Normal", "Retreat"),
            ("Honeypot", "Honeypot", "Attack"),
            ("Honeypot", "Honeypot", "Retreat")
        ]
        
        defender_rewards = []
        attacker_rewards = []
        
        # 真实系统数据分析
        for s in [0, 1]:  # 0=Normal, 1=Honeypot
            for a in [0, 1]:  # 0=Retreat, 1=Attack
                mask = (real_data[:, 0] == s) & (real_data[:, 1] == a)
                if np.any(mask):
                    defender_rewards.append(np.mean(real_data[mask, 2]))
                    attacker_rewards.append(np.mean(real_data[mask, 3]))
                else:
                    defender_rewards.append(0)
                    attacker_rewards.append(0)
        
        # 蜜罐系统数据分析
        for s in [0, 1]:  # 0=Normal, 1=Honeypot
            for a in [0, 1]:  # 0=Retreat, 1=Attack
                mask = (honeypot_data[:, 0] == s) & (honeypot_data[:, 1] == a)
                if np.any(mask):
                    defender_rewards.append(np.mean(honeypot_data[mask, 2]))
                    attacker_rewards.append(np.mean(honeypot_data[mask, 3]))
                else:
                    defender_rewards.append(0)
                    attacker_rewards.append(0)
        
        # 绘制策略效果图
        x = np.arange(len(categories))
        width = 0.35
        
        fig, ax = plt.subplots(figsize=(14, 8))
        ax.bar(x - width/2, defender_rewards, width, label='Defender', color='blue')
        ax.bar(x + width/2, attacker_rewards, width, label='Attacker', color='red')
        
        ax.set_ylabel('Average Reward')
        ax.set_title('Reward Analysis by System Type, Signal and Action')
        ax.set_xticks(x)
        ax.set_xticklabels([f"{c[0]}\n{c[1]}-{c[2]}" for c in categories], rotation=45)
        ax.legend()
        
        # 添加数值标签
        for i, v in enumerate(defender_rewards):
            ax.text(i - width/2, v + (0.5 if v >= 0 else -1.5), f'{v:.1f}', ha='center', fontsize=8)
        for i, v in enumerate(attacker_rewards):
            ax.text(i + width/2, v + (0.5 if v >= 0 else -1.5), f'{v:.1f}', ha='center', fontsize=8)
        
        plt.tight_layout()
        if save_dir:
            plt.savefig(os.path.join(save_dir, 'strategy_analysis.png'), dpi=300, bbox_inches='tight')
        plt.show()

def plot_history_analysis(history_data, save_dir=None):
    """绘制历史信息分析图表"""
    if save_dir and not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 创建子图
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 防御者历史影响
    ax1.plot(history_data["defender"]["signal_history_impact"], 
             label='Signal History Impact', color='blue')
    ax1.set_title('Defender Signal History Impact')
    ax1.set_xlabel('Steps')
    ax1.set_ylabel('Impact')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history_data["defender"]["action_history_impact"], 
             label='Action History Impact', color='green')
    ax2.set_title('Defender Action History Impact')
    ax2.set_xlabel('Steps')
    ax2.set_ylabel('Impact')
    ax2.legend()
    ax2.grid(True)
    
    # 攻击者历史影响
    ax3.plot(history_data["attacker"]["signal_history_impact"], 
             label='Signal History Impact', color='red')
    ax3.set_title('Attacker Signal History Impact')
    ax3.set_xlabel('Steps')
    ax3.set_ylabel('Impact')
    ax3.legend()
    ax3.grid(True)
    
    ax4.plot(history_data["attacker"]["action_history_impact"], 
             label='Action History Impact', color='orange')
    ax4.set_title('Attacker Action History Impact')
    ax4.set_xlabel('Steps')
    ax4.set_ylabel('Impact')
    ax4.legend()
    ax4.grid(True)
    
    plt.tight_layout()
    if save_dir:
        plt.savefig(os.path.join(save_dir, 'history_analysis.png'), dpi=300, bbox_inches='tight')
    plt.show()

def main(checkpoint_path, num_eval_episodes=100, results_dir="evaluation_results"):
    """主评估函数"""
    # 加载训练好的模型
    algo = load_algorithm(checkpoint_path)
    
    # 评估模型
    eval_results = evaluate_policy(algo, num_episodes=num_eval_episodes)
    
    # 创建结果目录
    os.makedirs(results_dir, exist_ok=True)
    
    # 保存评估结果
    with open(os.path.join(results_dir, 'eval_results.pkl'), 'wb') as f:
        pickle.dump(eval_results, f)
    
    # 创建评估结果表格
    results_table = pd.DataFrame({
        "Metric": ["Defender Reward", "Attacker Reward", "Episode Length", 
                  "Honeypot Signal Rate", "Attack Action Rate",
                  "Defender Signal History Impact", "Defender Action History Impact",
                  "Attacker Signal History Impact", "Attacker Action History Impact"],
        "Mean": [
            eval_results["defender_rewards"]["mean"],
            eval_results["attacker_rewards"]["mean"],
            eval_results["episode_lengths"]["mean"],
            eval_results["honeypot_signals"]["mean"],
            eval_results["attack_actions"]["mean"],
            np.mean(eval_results["history_analysis"]["defender"]["signal_history_impact"]),
            np.mean(eval_results["history_analysis"]["defender"]["action_history_impact"]),
            np.mean(eval_results["history_analysis"]["attacker"]["signal_history_impact"]),
            np.mean(eval_results["history_analysis"]["attacker"]["action_history_impact"])
        ],
        "Std": [
            eval_results["defender_rewards"]["std"],
            eval_results["attacker_rewards"]["std"],
            eval_results["episode_lengths"]["std"],
            eval_results["honeypot_signals"]["std"],
            eval_results["attack_actions"]["std"],
            np.std(eval_results["history_analysis"]["defender"]["signal_history_impact"]),
            np.std(eval_results["history_analysis"]["defender"]["action_history_impact"]),
            np.std(eval_results["history_analysis"]["attacker"]["signal_history_impact"]),
            np.std(eval_results["history_analysis"]["attacker"]["action_history_impact"])
        ]
    })
    
    # 保存表格
    results_table.to_csv(os.path.join(results_dir, 'evaluation_summary.csv'), index=False)
    print("评估摘要:")
    print(results_table)
    
    # 绘制评估结果图表
    plot_evaluation_results(eval_results, save_dir=results_dir)
    
    # 绘制历史分析图表
    plot_history_analysis(eval_results["history_analysis"], save_dir=results_dir)
    
    # 如果有训练历史，也可以绘制训练过程
    try:
        with open(os.path.join(os.path.dirname(checkpoint_path), 'training_results.pkl'), 'rb') as f:
            train_results = pickle.load(f)
            plot_rewards_over_time(train_results, save_path=os.path.join(results_dir, 'training_progress.png'))
    except:
        print("训练历史数据不可用，跳过绘制训练过程图。")
    
    print(f"评估完成！结果已保存到 {results_dir} 目录。")

if __name__ == "__main__":
    # 指定检查点路径
    CHECKPOINT_PATH = "mappo_signaling_checkpoints/checkpoint_000500"
    
    # 运行评估
    main(
        checkpoint_path=CHECKPOINT_PATH,
        num_eval_episodes=100,
        results_dir="evaluation_results"
    )