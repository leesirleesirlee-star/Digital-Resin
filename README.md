# Digital Resin

这是一个用于研究和演示材料模拟、扩散过程、孔隙生成以及 PINN（Physics-Informed Neural Network）求解的 Python 项目。

## 项目概览

该项目包含以下内容：

- 分子构建与初始结构生成：`build_monomer.py`
- 分子结构优化：`compute_binding_opt.py`
- 1D 扩散方程数值模拟：`diffusion1.py`
- 随机孔隙结构生成：`generate_pores.py`
- PINN 求解扩散方程：`pinn_diffusion.py`
- 吸附/扩散耦合问题中的 PINN 模型：`pinn_absorbtion.py`
- 简单原子距离计算示例：`hello_atoms.py`

## 目录说明

- `*.py`：核心脚本
- `*.xyz`：分子结构输入/输出文件
- `opt.traj`：ASE 优化轨迹文件

## 依赖环境

建议使用 Conda 环境，并安装以下依赖：

```bash
conda create -n digital-resin python=3.10 -y
conda activate digital-resin
pip install -r requirements.txt
```

## 运行示例

```bash
python build_monomer.py
python compute_binding_opt.py
python diffusion1.py
python generate_pores.py
python pinn_diffusion.py
python pinn_absorbtion.py
python hello_atoms.py
```

## 备注

- 这些脚本主要用于研究和教学场景，部分脚本依赖较重的科学计算库。
- 运行前请确保已安装 `numpy`、`matplotlib`、`ase`、`rdkit`、`deepxde` 和 `mace` 相关依赖。
- 生成的结果文件会保存在当前目录中，便于直接查看和后处理。

## 许可证

本项目为学习与研究用途，默认未指定商业许可证。若需要用于公开发布或商业用途，请在使用前确认权限。
