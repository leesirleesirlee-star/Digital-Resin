# Digital Twin of Ion Exchange Resin — MVP

![Status](https://img.shields.io/badge/status-MVP-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains the **minimum viable product (MVP)** of a multi-scale digital twin for ion exchange resins used in water treatment. The project simulates the adsorption of ions (e.g., Na⁺) onto a strong‑acid cation exchange resin (D001) by integrating:

- **Molecular modeling** (RDKit, ASE)
- **Machine‑learned interatomic potentials** (MACE‑MP‑0)
- **Pore structure generation** (Boolean model)
- **Physics‑Informed Neural Networks** (PINNs) for diffusion‑adsorption simulation

The MVP demonstrates a complete workflow from molecular structure to macroscopic adsorption prediction, laying the groundwork for future AI‑driven virtual experiments.

## Current Features

- Build a simplified resin monomer (styrene sulfonate) with a sodium ion using RDKit.
- Optimize the structure and compute binding energy using MACE‑MP‑0.
- Generate a 2D Boolean pore structure and estimate porosity.
- Solve a 1D diffusion‑adsorption equation with a Langmuir adsorption term using a physics‑informed neural network (DeepXDE).
- Extract adsorption kinetics (average adsorbed amount vs. time) and theoretical Langmuir isotherm.
- Calculate effective diffusivity from porosity via the Bruggeman relation.

## Project Structure

```
.
├── build_monomer.py              # RDKit: build styrene sulfonate + Na⁺
├── compute_binding_opt.py        # MACE‑MP‑0: optimize structure and compute binding energy
├── generate_pores.py             # Boolean model: generate 2D pore structure
├── pinn_adsorption.py            # DeepXDE: diffusion‑adsorption PINN (core) with kinetics extraction
├── main.py                       # (planned) unified entry point for parameter sweeps
├── styrene_sulfonate_na.xyz      # initial molecular structure
├── styrene_sulfonate_na_opt.xyz  # optimized structure
└── README.md
```

## Installation

Create a dedicated conda environment:

```bash
conda create -n resin_dt python=3.10
conda activate resin_dt
```

Install required packages:

```bash
pip install torch torchvision torchaudio
pip install ase rdkit numpy matplotlib pandas
pip install deepxde
pip install mace-torch
```

> **Note:** `mace-torch` will download the MACE‑MP‑0 model weights (~500 MB) on first use. For users in restricted network environments, consider using a mirror such as `https://pypi.tuna.tsinghua.edu.cn/simple`.

## Usage

### 1. Build the resin monomer + ion structure

```bash
python build_monomer.py
```

Generates `styrene_sulfonate_na.xyz`.

### 2. Optimize and compute binding energy

```bash
python compute_binding_opt.py
```

Uses MACE‑MP‑0 (CPU) to relax the structure and outputs the binding energy (e.g., −264.6 kJ/mol for Na⁺).

### 3. Generate pore structure

```bash
python generate_pores.py
```

Creates a 2D Boolean pore network with a porosity of ~0.343. Parameters can be changed inside the script.

### 4. Run the diffusion‑adsorption PINN and extract kinetics

```bash
python pinn_adsorption.py
```

This script:
- Computes the effective diffusivity from porosity using the Bruggeman relation.
- Trains a PINN to solve the 1D diffusion‑adsorption equation.
- Plots the average adsorbed amount versus time (kinetics) and concentration profiles at selected times.
- Prints the theoretical equilibrium adsorbed amount for comparison.

## Expected Output

- `build_monomer.py`: prints atom information and saves the XYZ file.
- `compute_binding_opt.py`: prints binding energy in eV and kJ/mol.
- `generate_pores.py`: displays a 2D pore structure image.
- `pinn_adsorption.py`: shows two plots — adsorption kinetics and concentration profiles — and prints the simulated final average adsorption amount versus the theoretical value.

## Physical Model & Simplifications

The current MVP relies on several simplifications:

- The resin is represented by a single styrene sulfonate monomer (not a polymer network).
- Only one ion (Na⁺) is considered.
- The pore structure is a 2D Boolean model (single‑scale).
- Adsorption is described by the Langmuir isotherm under local equilibrium.
- The diffusion coefficient and adsorption parameters are dimensionless; they are mapped from porosity but not yet fully calibrated to experimental data.
- Temperature effects are not yet included.

These simplifications will be progressively replaced by more realistic multi‑ion, multi‑scale models.

## Roadmap

- [ ] Extract adsorption isotherms directly from PINN equilibrium results
- [ ] Integrate all modules into a single `main.py` with user‑defined parameters
- [ ] Support multiple ions (Ca²⁺, Mg²⁺, heavy metals) and temperatures
- [ ] Add a fixed‑bed column model for breakthrough curves
- [ ] Add regeneration behavior simulation
- [ ] Perform parameter sensitivity analysis
- [ ] Build a Streamlit web interface for interactive prediction
- [ ] Implement active learning to incorporate experimental data

## License

This project is licensed under the MIT License.

## Acknowledgements

- [MACE‑MP‑0](https://github.com/ACEsuit/mace-mp) for machine‑learned interatomic potentials
- [DeepXDE](https://github.com/lululxvi/deepxde) for physics‑informed neural networks
- [RDKit](https://www.rdkit.org/) for molecular construction
- [ASE](https://wiki.fysik.dtu.dk/ase/) for atomistic operations

## Contact

For questions or collaboration, please contact [leesirleesirlee@gmail.com].