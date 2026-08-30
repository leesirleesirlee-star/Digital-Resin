import numpy as np
import matplotlib.pyplot as plt

def generate_boolean_pores(grid_size=100, porosity=0.4, sphere_radius=5, seed=42):
    rng = np.random.default_rng(seed)
    volume = np.ones((grid_size, grid_size), dtype=np.uint8)  # 初始全为固体(1)
    
    # 随机放置球体（圆）作为孔隙，将孔隙区域设为0
    n_spheres = int(porosity * grid_size**2 / (np.pi * sphere_radius**2))
    for _ in range(n_spheres):
        cx = rng.integers(0, grid_size)
        cy = rng.integers(0, grid_size)
        y, x = np.ogrid[:grid_size, :grid_size]
        mask = (x - cx)**2 + (y - cy)**2 <= sphere_radius**2
        volume[mask] = 0  # 孔隙=0，固体=1
    
    actual_porosity = 1 - volume.mean()
    return volume, actual_porosity

volume, por = generate_boolean_pores(grid_size=100, porosity=0.4, sphere_radius=5)
print(f"实际孔隙率: {por:.3f}")

plt.imshow(volume, cmap='gray', origin='lower')
plt.title(f"Boolean pore structure (porosity={por:.3f})")
plt.colorbar()
plt.show()