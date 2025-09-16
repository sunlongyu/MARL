import matplotlib.pyplot as plt
from typing import List
import pandas as pd
import numpy as np
plt.rcParams['font.family'] = 'Times New Roman'
# 添加必要的常量
DEFENDER = "defender"
ATTACKER = "attacker"

# 添加策略映射函数
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id == DEFENDER:
        return "defender_policy"
    elif agent_id == ATTACKER:
        return "attacker_policy"
    else:
        raise ValueError(f"未知智能体ID: {agent_id}")
class TrainingTracker:
    def __init__(self):
        self.defender_rewards = []
        self.attacker_rewards = []
        self.defender_policy_loss = []  # 修改这里
        self.attacker_policy_loss = []  # 修改这里
        self.episodes = []
        
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
        def_loss = metrics.get("defender_policy", {}).get("policy_loss", float('nan'))
        att_loss = metrics.get("attacker_policy", {}).get("policy_loss", float('nan'))
        self.defender_policy_loss.append(def_loss)  # 修改这里
        self.attacker_policy_loss.append(att_loss)  # 修改这里
        
        self.episodes.append(iteration)


def load_tracker_from_file(checkpoint_dir_path):
    """从文件加载训练跟踪器"""
    #保存tracker为pkl文件
    import pickle
    with open(checkpoint_dir_path+"/training_tracker.pkl", "rb") as f:
        tracker = pickle.load(f)
    return tracker

def plot_training_data(tracker_list,learn_rate_list,save_path="./figures",window=10,font_size=25):
    """画图
    tracker_list: list of tracker in different learning rate
    """
    #刻画不同学习率下的策略损失
    plot_attcker_loss(tracker_list,learn_rate_list,save_path,window=window,font_size=font_size)
    plot_defender_loss(tracker_list,learn_rate_list,save_path,window=window,font_size=font_size)
    #刻画不同学习率下的策略奖励
    plot_attacker_reward(tracker_list,learn_rate_list,save_path,window=window,font_size=font_size)
    plot_defender_reward(tracker_list,learn_rate_list,save_path,window=window,font_size=font_size)

def get_learn_rate_str(num):
    """将学习率转换为字符串，返回LaTeX格式"""
    # 将学习率转换为科学计数法，然后格式化为LaTeX上标形式
    exponent = int(np.floor(np.log10(abs(num))))
    mantissa = num / (10 ** exponent)
    return rf"${mantissa:.0f}\times10^{{{exponent}}}$"

def plot_attcker_loss(tracker_list, learn_rate_list, save_path="./figures", window=5, font_size=12):
    """绘制不同学习率下的攻击者策略损失曲线
    
    Args:
        tracker_list: 不同学习率下的跟踪器对象列表
        learn_rate_list: 与跟踪器对应的学习率列表
        font_size: 图表字体大小
    """
    plt.figure(figsize=(10, 8))
    
    # 设置字体大小
    plt.rcParams.update({'font.size': font_size})
    
    for i, (tracker, lr) in enumerate(zip(tracker_list, learn_rate_list)):
        # Plot loss curve with thicker lines
        plt.plot(tracker.episodes, tracker.attacker_policy_loss, label=f'lr={get_learn_rate_str(lr)}', 
                linewidth=2.5)
        
        # Add standard deviation area
        loss_std = pd.Series(tracker.attacker_policy_loss).rolling(window=window).std()
        
        plt.fill_between(tracker.episodes,
                        [l - s for l, s in zip(tracker.attacker_policy_loss, loss_std)],
                        [l + s for l, s in zip(tracker.attacker_policy_loss, loss_std)],
                        alpha=0.2)
    
    plt.xlabel('Training Episodes', fontsize=font_size)
    plt.ylabel('Attacker Policy Loss', fontsize=font_size)
    # plt.title('Attacker Policy Loss for Different Learning Rates', fontsize=font_size)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    # Adjust x-axis tick range and format tick labels with unit 'K'
    
    # Save figures
    plt.tight_layout()
    plt.savefig(save_path+"/attacker_loss.png", dpi=300)
    plt.savefig(save_path+"/attacker_loss.pdf")
    plt.close()

def plot_defender_loss(tracker_list, learn_rate_list, save_path="./figures", window=5, font_size=25):
    """绘制不同学习率下的防御者策略损失曲线
    
    Args:
        tracker_list: 不同学习率下的跟踪器对象列表
        learn_rate_list: 与跟踪器对应的学习率列表
        font_size: 图表字体大小
    """
    plt.figure(figsize=(10, 8))
    
    # 设置字体大小
    plt.rcParams.update({'font.size': font_size})
    
    for i, (tracker, lr) in enumerate(zip(tracker_list, learn_rate_list)):
        # Plot loss curve with thicker lines
        plt.plot(tracker.episodes, tracker.defender_policy_loss, label=f'lr={get_learn_rate_str(lr)}',
                linewidth=2.5)
        
        # Add standard deviation area
        loss_std = pd.Series(tracker.defender_policy_loss).rolling(window=window).std()
        
        plt.fill_between(tracker.episodes,
                        [l - s for l, s in zip(tracker.defender_policy_loss, loss_std)],
                        [l + s for l, s in zip(tracker.defender_policy_loss, loss_std)],
                        alpha=0.2)
    
    plt.xlabel('Training Episodes', fontsize=font_size)
    plt.ylabel('Defender Policy Loss', fontsize=font_size)
    # plt.title('Defender Policy Loss for Different Learning Rates', fontsize=font_size)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    # Adjust x-axis tick range and format tick labels with unit 'K'
    
    # Save figures
    plt.tight_layout()
    plt.savefig(save_path+"/defender_loss.png", dpi=300)
    plt.savefig(save_path+"/defender_loss.pdf")
    plt.close()

def plot_attacker_reward(tracker_list, learn_rate_list, save_path="./figures", window=5, font_size=12):
    """绘制不同学习率下的攻击者奖励曲线
    
    Args:
        tracker_list: 不同学习率下的跟踪器对象列表
        learn_rate_list: 与跟踪器对应的学习率列表
        font_size: 图表字体大小
    """
    plt.figure(figsize=(10, 8))
    
    # 设置字体大小
    plt.rcParams.update({'font.size': font_size})
    
    for i, (tracker, lr) in enumerate(zip(tracker_list, learn_rate_list)):
        # Plot reward curve with thicker lines
        plt.plot(tracker.episodes, tracker.attacker_rewards, label=f'lr={get_learn_rate_str(lr)}',
                linewidth=2.5)
        
        # Add standard deviation area
        reward_std = pd.Series(tracker.attacker_rewards).rolling(window=window).std()
        
        plt.fill_between(tracker.episodes,
                        [r - s for r, s in zip(tracker.attacker_rewards, reward_std)],
                        [r + s for r, s in zip(tracker.attacker_rewards, reward_std)],
                        alpha=0.2)
    
    plt.xlabel('Training Episodes', fontsize=font_size)
    plt.ylabel('Attacker Reward', fontsize=font_size)
    # Adjust x-axis tick range and format tick labels with unit 'K'
    
    # plt.title('Attacker Reward for Different Learning Rates', fontsize=font_size)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
   
    
    # Save figures
    plt.tight_layout()
    plt.savefig(save_path+"/attacker_reward.png", dpi=300)
    plt.savefig(save_path+"/attacker_reward.pdf")
    plt.close()

def plot_defender_reward(tracker_list, learn_rate_list, save_path="./figures", window=5, font_size=25):
    """绘制不同学习率下的防御者奖励曲线
    
    Args:
        tracker_list: 不同学习率下的跟踪器对象列表
        learn_rate_list: 与跟踪器对应的学习率列表
        font_size: 图表字体大小
    """
    plt.figure(figsize=(10, 8))
    
    # 设置字体大小
    plt.rcParams.update({'font.size': font_size})
    
    for i, (tracker, lr) in enumerate(zip(tracker_list, learn_rate_list)):
        # Plot reward curve with thicker lines
        plt.plot(tracker.episodes, [num+100 for num in tracker.defender_rewards], label=f'lr={get_learn_rate_str(lr)}',
                linewidth=2.5)
        
        # Add standard deviation area
        reward_std = pd.Series([num+100 for num in tracker.defender_rewards]).rolling(window=window).std()
        
        plt.fill_between(tracker.episodes,
                        [r - s for r, s in zip([num+100 for num in tracker.defender_rewards], reward_std)],
                        [r + s for r, s in zip([num+100 for num in tracker.defender_rewards], reward_std)],
                        alpha=0.2)
    
    plt.xlabel('Training Episodes', fontsize=font_size)
    plt.ylabel('Defender Reward', fontsize=font_size)
    # plt.title('Defender Reward for Different Learning Rates', fontsize=font_size)
    plt.legend(fontsize=font_size)
    # plt.grid(True)
    # Adjust x-axis tick range and format tick labels with unit 'K'
  
    # Save figures
    plt.tight_layout()
    plt.savefig(save_path+"/defender_reward.png", dpi=300)
    plt.savefig(save_path+"/defender_reward.pdf")
    plt.close()

if __name__ == "__main__":
    #从文件加载训练跟踪器
    learn_rate_name_list = ["lr1","lr2","lr3","lr4"]
    learn_rate_list = [2e-4, 1e-4, 7e-5, 5e-5]
    tracker_list = []
    for name in learn_rate_name_list:
        checkpoint_dir_path = f"./checkpoints_{name}"
        tracker = load_tracker_from_file(checkpoint_dir_path)
        tracker_list.append(tracker)

    
    plot_training_data(tracker_list,learn_rate_list)