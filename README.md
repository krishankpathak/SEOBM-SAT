# SEOBM-SAT

This repository contains the reference implementation for the paper:

**SEOBM: A Temporal Representation for Learning-Based Orbital Behavior Modeling**

## Overview
The Self-Evolving Orbital Behavior Map (SEOBM) is a structured temporal representation that encodes physically interpretable orbital behavior features over a finite memory window. The goal of this repository is to provide a reproducible, controlled evaluation of temporal representations for learning-based orbital analysis.

This codebase is **not** an operational SSA system and makes **no claims** regarding collision avoidance, satellite control, or real-world deployment.

----

## Repository Structure

- `src/` — Core implementation (parsing, physics, features, models, RL)
- `notebooks/` — Reproducible experiment notebooks and figure generation
- `data/` — Public anchor data and synthetic trajectories (see notes below)
- `figures/` — Final figures used in the paper
- `paper/` — Compiled manuscript and submission figures

---

## Installation

```bash
git clone https://github.com/<username>/SEOBM-SAT.git
cd SEOBM-SAT
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows
pip install -r requirements.txt
