import os
import time
import numpy as np
import ray
from ray.rllib.algorithms.ppo import PPO
from ray.rllib.policy.policy import PolicySpec
from ray.rllib.env import PettingZooEnv
from ray.rllib.env.wrappers.multi_agent_env_compatibility import MultiAgentEnvCompatibility
from ray.tune.registry import register_env
import torch

# 导入环境
from env import parallel_env, DEFENDER, ATTACKER, MAX_STEPS, N_SYSTEMS

# 环境注册与配置 - 添加兼容性包装器
def create_rllib_env(config):
    """创建Ray兼容的环境包装器, 使用兼容性包装器处理API差异"""
    # 从config中获取参数，如果未指定则使用默认值
    n_systems = config.get("n_systems", N_SYSTEMS)
    max_steps = config.get("max_steps", MAX_STEPS)
    history_length = config.get("history_length", 5)
    
    env = parallel_env(N=n_systems, T=max_steps, history_length=history_length)
    
    # 使用PettingZooEnv包装
    env = PettingZooEnv(env)
    
    # 关键修改: 使用MultiAgentEnvCompatibility包装使其兼容Gymnasium API
    env = MultiAgentEnvCompatibility(env)
    
    # 打印观察空间信息以便调试
    print(f"\n观察空间信息:")
    for agent_id, space in env.observation_space.spaces.items():
        print(f"智能体 {agent_id} 的观察空间: {space} (类型: {type(space).__name__})")
    
    return env

# 策略映射函数
def policy_mapping_fn(agent_id, episode=None, worker=None, **kwargs):
    if agent_id == DEFENDER:
        return "defender_policy"
    elif agent_id == ATTACKER:
        return "attacker_policy"
    else:
        raise ValueError(f"未知智能体ID: {agent_id}")

# 训练函数 - 适配最新Ray API
def run_mappo_training(
    stop_iters=200, 
    checkpoint_path="mappo_checkpoints", 
    use_lstm=True, 
    lstm_cell_size=64, 
    seed=42, 
    num_workers=0,
    n_systems=N_SYSTEMS, 
    max_steps=MAX_STEPS, 
    history_length=5,
    use_tensorboard=True
):
    """使用MAPPO训练防御者和攻击者策略 - 适配最新Ray RLlib版本"""
    print("初始化Ray...")
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)
    print(f"Ray已初始化: {ray.is_initialized()}")
    print(f"Ray可用资源: {ray.available_resources()}")
    
    # 注册环境
    env_name = "multistage_signaling_v0"
    register_env(env_name, create_rllib_env)
    
    # 创建临时环境实例以获取观察和动作空间
    temp_env = create_rllib_env({
        "n_systems": n_systems,
        "max_steps": max_steps,
        "history_length": history_length
    })
    obs_spaces = temp_env.observation_space
    act_spaces = temp_env.action_space
    temp_env.close()
    
    # 使用新的 Ray RLlib API 配置
    config = {
        # 环境配置
        "env": env_name,
        "env_config": {
            "n_systems": n_systems,
            "max_steps": max_steps,
            "history_length": history_length
        },
        
        # 框架选择
        "framework": "torch",
        
        # 关键添加: 禁用新的API栈以避免兼容性问题
        "_disable_rl_module_and_learner": True,
        "_disable_env_runner_and_connector_v2": True,
        
        # 多智能体配置
        "multiagent": {
            "policies": {
                "defender_policy": PolicySpec(
                    observation_space=obs_spaces[DEFENDER],
                    action_space=act_spaces[DEFENDER],
                    config={"agent_id": DEFENDER},
                ),
                "attacker_policy": PolicySpec(
                    observation_space=obs_spaces[ATTACKER],
                    action_space=act_spaces[ATTACKER],
                    config={"agent_id": ATTACKER},
                ),
            },
            "policy_mapping_fn": policy_mapping_fn,
            "policies_to_train": ["defender_policy", "attacker_policy"],
        },
        
        # 训练参数 - 基本必需参数
        "gamma": 0.99,
        "lr": 3e-4,
        "train_batch_size": 4000,
        
        # worker参数 - 使用旧参数名称以兼容禁用的新API栈
        "num_workers": num_workers,
        "rollout_fragment_length": 200,
        
        # 模型配置
        "model": {
            "fcnet_hiddens": [128, 128],
            "use_lstm": use_lstm,
            "lstm_cell_size": lstm_cell_size,
        },
        
        # 随机种子
        "seed": seed,
        
        # 日志设置
        "log_level": "INFO",
    }
    
    # 创建算法
    algo = PPO(config=config)
    print("MAPPO算法创建成功!")
    
    # 创建检查点目录
    os.makedirs(checkpoint_path, exist_ok=True)
    print(f"检查点将保存至: {checkpoint_path}")
    
    # 开始训练
    print(f"开始训练，共{stop_iters}轮迭代...")
    start_time = time.time()
    
    train_results = []
    for i in range(stop_iters):
        iter_start = time.time()
        result = algo.train()
        iter_time = time.time() - iter_start
        
        # 记录结果
        train_results.append(result)
        
        # 输出关键指标
        print(f"\n===== 迭代 {i+1}/{stop_iters}, 用时: {iter_time:.2f}秒 =====")
        
        # 输出详细智能体奖励
        rewards = result.get("policy_reward_mean", {})
        def_reward = rewards.get("defender_policy", float('nan'))
        att_reward = rewards.get("attacker_policy", float('nan'))
        print(f"平均奖励: 防御者={def_reward:.2f}, 攻击者={att_reward:.2f}")
        
        # 每N轮保存一次检查点
        checkpoint_interval = min(50, max(10, stop_iters // 10))
        if (i+1) % checkpoint_interval == 0 or (i+1) == stop_iters:
            checkpoint = algo.save(checkpoint_path)
            print(f"检查点已保存: {checkpoint}")
        
        print("=" * 50)
    
    # 训练完成，保存最终检查点
    final_checkpoint = algo.save(checkpoint_path)
    print(f"训练完成，总用时: {(time.time()-start_time)/60:.2f}分钟")
    print(f"最终检查点: {final_checkpoint}")
    
    # 分析训练结果
    analyze_training_results(train_results)
    
    # 清理
    algo.stop()
    if ray.is_initialized():
        ray.shutdown()
    
    return train_results, final_checkpoint

def analyze_training_results(results):
    """分析并打印训练结果的关键统计数据"""
    if not results:
        print("没有可用的训练结果进行分析")
        return
    
    # 提取各策略的奖励
    defender_rewards = []
    attacker_rewards = []
    
    for result in results:
        if "policy_reward_mean" in result:
            rewards = result["policy_reward_mean"]
            if "defender_policy" in rewards:
                defender_rewards.append(rewards["defender_policy"])
            if "attacker_policy" in rewards:
                attacker_rewards.append(rewards["attacker_policy"])
    
    if defender_rewards and attacker_rewards:
        # 计算最终奖励
        final_def_reward = defender_rewards[-1]
        final_att_reward = attacker_rewards[-1]
        
        # 计算趋势
        def_trend = "上升" if defender_rewards[-1] > defender_rewards[0] else "下降"
        att_trend = "上升" if attacker_rewards[-1] > attacker_rewards[0] else "下降"
        
        print("\n===== 训练结果分析 =====")
        print(f"训练轮次: {len(results)}")
        print(f"防御者最终平均奖励: {final_def_reward:.2f} (趋势: {def_trend})")
        print(f"攻击者最终平均奖励: {final_att_reward:.2f} (趋势: {att_trend})")
        print(f"防御者vs攻击者: {'防御者领先' if final_def_reward > final_att_reward else '攻击者领先'}")
        print("========================\n")

if __name__ == "__main__":
    # 训练参数设置
    TRAINING_ITERATIONS = 500
    USE_LSTM = True
    LSTM_CELL_SIZE = 128
    SEED = 42
    CHECKPOINT_PATH = "mappo_signaling_checkpoints"
    
    # 环境参数
    N_SYSTEMS = 3
    MAX_STEPS = 100
    HISTORY_LENGTH = 5
    
    # 运行训练
    results, checkpoint = run_mappo_training(
        stop_iters=TRAINING_ITERATIONS,
        checkpoint_path=CHECKPOINT_PATH,
        use_lstm=USE_LSTM,
        lstm_cell_size=LSTM_CELL_SIZE,
        seed=SEED,
        num_workers=0,  # 根据可用CPU资源适当调整
        n_systems=N_SYSTEMS,
        max_steps=MAX_STEPS,
        history_length=HISTORY_LENGTH,
        use_tensorboard=True  # 启用TensorBoard可视化
    )
    
    print(f"训练完成！已保存检查点: {checkpoint}")
    print("可以使用以下命令加载模型进行评估:")
    print(f"from ray.rllib.algorithms.ppo import PPO")
    print(f"algo = PPO.from_checkpoint('{checkpoint}')")
    print(f"# 然后使用algo.evaluate()或自定义评估代码")