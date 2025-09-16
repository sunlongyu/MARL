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
N_SYSTEMS_DEFAULT = 3       # Default number of systems 系统主机数量
MAX_STEPS_DEFAULT = 200     # Default max steps (T horizon) Agent最大攻防步数
HISTORY_LEN_DEFAULT = 50     # Default history length for observation 最大观测长度记忆

# --- Payoffs (CRITICAL: Verify these match your Table 1 exactly!) ---
R_A = 2.0   # 增加攻击真实系统的收益
C_A = 1.5   # 保持攻击成本不变
L_A = 3.0   # 增加防御损失，与攻击收益对称
L_I = 4.0   # 降低蜜罐收益，避免过度威慑
C_N = 2.0   # 增加欺骗成本，抑制过度使用
C_H = 1.5   # 适当增加蜜罐维护成本
A_R = 2.0   # 增加识破奖励
D_R = 3.0   # 增加成功欺骗奖励
A_D = 0.5   # 降低撤退收益，鼓励交互

# ==============================================================================