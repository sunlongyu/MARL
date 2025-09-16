# ==============================================================================
# === Constants and Payoffs ===
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
N_SYSTEMS_DEFAULT = 4       # 系统主机数量
MAX_STEPS_DEFAULT = 25      # Agent最大攻防步数 
HISTORY_LEN_DEFAULT = 8     # 最大观测长度记忆

# --- Payoffs ---
R_A = 2.5   # 攻击真实系统的收益
C_A = 1.8   # 攻击成本
L_A = 3.5   # 防御损失
L_I = 4.5   # 蜜罐收益
C_N = 2.2   # 欺骗成本
C_H = 1.8   # 蜜罐维护成本
A_R = 2.2   # 识破奖励
D_R = 3.2   # 成功欺骗奖励
A_D = 0.8   # 撤退收益

# ==============================================================================
