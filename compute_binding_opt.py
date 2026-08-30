from ase.io import read
from ase.optimize import BFGS
from mace.calculators import mace_mp

# 读取初始结构
atoms = read('styrene_sulfonate_na.xyz')

# 使用 MACE-MP-0 计算器（CPU）
calc = mace_mp(model="medium", device="cpu")
atoms.calc = calc

# 结构优化
opt = BFGS(atoms, trajectory='opt.traj')
opt.run(fmax=0.05)  # 力收敛阈值

# 保存优化后的结构
atoms.write('styrene_sulfonate_na_opt.xyz')
print("优化完成，结构已保存为 styrene_sulfonate_na_opt.xyz")

# 计算复合物能量
E_complex = atoms.get_potential_energy()

# 删除 Na 得到树脂片段
resin_atoms = atoms.copy()
del resin_atoms[[atom.index for atom in resin_atoms if atom.symbol == 'Na']]
resin_atoms.calc = calc
E_resin = resin_atoms.get_potential_energy()

# 单独 Na
na_atoms = atoms[[atom.index for atom in atoms if atom.symbol == 'Na']]
na_atoms.calc = calc
E_na = na_atoms.get_potential_energy()

E_bind = E_complex - E_resin - E_na
print(f"优化后复合物能量: {E_complex:.6f} eV")
print(f"优化后树脂片段能量: {E_resin:.6f} eV")
print(f"Na+能量: {E_na:.6f} eV")
print(f"结合能: {E_bind:.6f} eV = {E_bind * 96.485:.3f} kJ/mol")