import numpy as np

# 定义三个原子坐标（单位：Å)
positions = np.array([
    [0.0,0.0,0.0], # atom1: S
    [1.5,0.0,0.0], # atom2: O
    [0.0,1.5,0.0], # atom3: Na
])

# 计算atom1和atom2之间的距离
distance = np.linalg.norm(positions[0]-positions[1])
print(f"S-O距离: {distance: .2f}Å")

# 计算所有原子对之间的距离
for i in range(len(positions)):
    for j in range(i+1, len(positions)):
        d = np.linalg.norm(positions[i]-positions[j])
        print(f"atom{i+1}-atom{j+1}距离: {d: .2f}Å")
