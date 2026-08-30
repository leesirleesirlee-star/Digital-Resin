# Digital Twin of Ion Exchange Resin — MVP

![Status](https://img.shields.io/badge/status-MVP-orange)
![Python](https://img.shields.io/badge/Python-3.10-blue)
![License](https://img.shields.io/badge/license-MIT-green)

## Overview

This repository contains the **minimum viable product (MVP)** of a multi-scale digital twin for ion exchange resins used in water treatment. The project aims to simulate the adsorption behavior of ions (e.g., Na⁺) on a strong-acid cation exchange resin (D001) by integrating:

- **Molecular modeling** (RDKit, ASE)
- **Machine-learned interatomic potentials** (MACE‑MP‑0)
- **Pore structure generation** (Boolean model)
- **Physics-Informed Neural Networks** (PINNs) for diffusion-adsorption simulation

The MVP demonstrates a complete workflow from molecular structure to macroscopic adsorption prediction, laying the foundation for future AI‑driven virtual experiments.

## Current Features

- Build a simplified resin monomer (styrene sulfonate) with a sodium ion.
- Optimize the structure and compute binding energy using MACE‑MP‑0.
- Generate a 2D Boolean pore structure and estimate porosity.
- Solve a 1D diffusion-adsorption equation with a Langmuir adsorption term using a physics-informed neural network (DeepXDE).
- Visualize concentration profiles and loss history.

## Project Structure
├── build_monomer.py # RDKit: build styrene sulfonate + Na⁺
├── compute_binding_opt.py # MACE‑MP‑0: optimize structure and compute binding energy
├── generate_pores.py # Boolean model: generate 2D pore structure
├── pinn_diffusion.py # DeepXDE: pure diffusion PINN (reference)
├── pinn_adsorption.py # DeepXDE: diffusion-adsorption PINN (core)
├── styrene_sulfonate_na.xyz # initial molecular structure
├── styrene_sulfonate_na_opt.xyz # optimized structure
└── README.md
