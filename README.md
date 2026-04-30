# Comparative Study of PINNs, XPINNs, and Quantum-Assisted PINNs for Burgers and Poisson Equations

This repository contains the final notebook, report, source code, trained model outputs, and result figures for a machine learning physics project comparing classical PINNs, XPINNs, and quantum-assisted PINNs on two PDE benchmarks: the 1D viscous Burgers equation and the 2D Poisson equation.

## Main Files

- `ML_for_Physics_Project.ipynb` - combined project notebook (reproducible on collab)
- `ML_Project_final.pdf` - final project report

## Project Summary

The project compares five models for each PDE benchmark:

- Classical PINN
- XPINN with first domain split
- XPINN with second domain split
- Shallow QA-PINN
- Deep QA-PINN

For the Burgers equation, the space-split XPINN gives the best relative L2 error.  
For the Poisson equation, the shallow QA-PINN gives the best relative L2 error.


