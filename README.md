# SEOBM: Self-Evolving Orbital Behavior Map

This repository contains the research code accompanying the paper:

**“SEOBM: A Temporal Representation for Learning-Based Orbital Behavior Modeling”**
by **Krishan Kant Pathak and Subodh Kushwaha**

The project explores a representation learning approach for modeling orbital dynamics over time. The core contribution is the **Self-Evolving Orbital Behavior Map (SEOBM)** — a structured temporal representation that captures orbital evolution using physically interpretable features.

---

# Project Motivation

Space Situational Awareness (SSA) increasingly involves analyzing complex orbital behavior in environments with growing satellite density. Traditional orbital mechanics provides accurate physical models, but integrating these dynamics into **learning-based systems** remains challenging.

Most machine learning pipelines rely on instantaneous orbital state vectors such as:

* Cartesian position and velocity
* Orbital elements
* Short state sequences

While effective for local predictions, these representations often fail to capture **long-term temporal behavior** required for:

* delayed prediction
* maneuver pattern recognition
* multi-agent coordination tasks

SEOBM addresses this by introducing a **fixed-dimension temporal tensor representation** derived from physically interpretable orbital features.

---

# Self-Evolving Orbital Behavior Map (SEOBM)

SEOBM converts orbital state vectors into a temporal representation:

```
SEOBM(t) ∈ ℝ^(F × K)
```

Where:

* **F** = number of behavior features
* **K** = memory window size

Each column corresponds to a feature vector derived from an orbital state.

The feature vector used in this work includes:

* orbital speed
* radial distance
* angular momentum magnitude
* radial velocity component

These features are computed deterministically from Cartesian states.

The representation evolves through a **sliding memory window**, preserving recent orbital behavior.

---

# Repository Contents

### src/

Core implementation of the SEOBM representation and learning pipelines.

Modules include:

* feature extraction
* SEOBM construction
* normalization
* baseline prediction models
* reinforcement learning integration

---

### notebooks/

Exploratory notebooks for:

* trajectory generation
* SEOBM tensor visualization
* experiment analysis

These notebooks help reproduce figures and experimental insights from the paper.

---

### scripts/

Utility scripts for running experiments and generating synthetic orbital trajectories.

---

### docs/

Supporting documentation summarizing methodology and experimental design.

---

# Dataset Policy

The repository **does not include the datasets used in experiments**.

This decision is intentional and follows the methodology described in the paper.

The experimental pipeline uses a **two-layer data strategy**:

### Layer-2: Real-World Anchor Data

Publicly available Starlink satellite geodetic data is used only to define realistic orbital regimes such as altitude and inclination ranges.
It is **not used for model training or evaluation**. 

### Layer-1: Synthetic Trajectory Generation

All training and evaluation experiments rely on **synthetic orbital trajectories generated through simplified propagation models**. 

Synthetic generation ensures:

* full control of ground truth
* reproducible experiments
* isolation of representation effects

---

# Why the `data/` Folder Is Empty

The `data/` directory is included only as a placeholder.

Real datasets are not stored in this repository because:

* large datasets exceed Git repository limits
* experiments rely primarily on synthetic trajectories
* real-world satellite data is used only as a reference for orbital regimes

Instructions for regenerating trajectories are provided in the scripts directory.

---

# Reproducing Experiments

Install dependencies:

```
pip install -r requirements.txt
```

Generate synthetic trajectories:

```
python scripts/generate_synthetic_orbits.py
```

Run supervised learning experiments:

```
python scripts/run_experiments.py
```

---

# Experimental Scope

All experiments in this repository are designed for **representation learning research**.

The work **does not propose**:

* operational SSA systems
* collision avoidance algorithms
* satellite control policies

Results should be interpreted only within controlled experimental settings.

---

# Research Scope

SEOBM should be viewed as a **representational abstraction** for learning-based orbital analysis rather than a deployable system.

The repository is intended to provide:

* reproducible experiments
* a structured temporal representation
* a foundation for further research at the intersection of astrodynamics and machine learning.

---

# License

This project is released for research and academic use.
