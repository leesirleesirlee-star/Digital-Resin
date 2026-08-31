import numpy as np
import matplotlib.pyplot as plt

q_max = 4.5
K = 10.0

C_e = np.linspace(0.01, 10, 100)  # 平衡浓度范围
q_e = q_max * K * C_e / (1 + K * C_e)

plt.figure(figsize=(8, 5))
plt.plot(C_e, q_e, linewidth=2)
plt.xlabel("Equilibrium concentration C_e")
plt.ylabel("Equilibrium adsorbed amount q_e")
plt.title("Langmuir adsorption isotherm")
plt.grid(True, alpha=0.3)
plt.show()