from rdkit import Chem
from rdkit.Chem import AllChem
from ase import Atoms
from ase.io import write

# 苯乙烯磺酸阴离子（用乙基代替乙烯基，简化模型）
# SMILES: 乙基苯磺酸阴离子
smiles = "CCc1ccc(cc1)S(=O)(=O)[O-]"
mol = Chem.MolFromSmiles(smiles)
mol = Chem.AddHs(mol)

# 生成三维坐标并用分子力场做初步优化
AllChem.EmbedMolecule(mol, randomSeed=42)
AllChem.MMFFOptimizeMolecule(mol)

# 提取坐标和元素符号
conf = mol.GetConformer()
positions = []
symbols = []
for atom in mol.GetAtoms():
    pos = conf.GetAtomPosition(atom.GetIdx())
    positions.append([pos.x, pos.y, pos.z])
    symbols.append(atom.GetSymbol())

# 转换为 ASE Atoms 对象
atoms = Atoms(symbols=symbols, positions=positions)

# 找到磺酸基上的氧原子，在其附近放置钠离子
o_indices = [i for i, s in enumerate(symbols) if s == 'O']
if o_indices:
    o_pos = positions[o_indices[0]]
    na_pos = [o_pos[0] + 2.0, o_pos[1], o_pos[2]]
    atoms += Atoms('Na', positions=[na_pos])

# 添加真空层，避免周期性镜像干扰
atoms.center(vacuum=8.0)

# 保存结构
write('styrene_sulfonate_na.xyz', atoms)
print("结构已保存为 styrene_sulfonate_na.xyz")
print(atoms)