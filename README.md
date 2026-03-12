# SEOBM: Self-Evolving Orbital Behavior Map

**A Temporal Representation for Learning-Based Orbital Behavior Modeling**

Author: **Krishan Kant Pathak**
CO-Author: **Subodh Kushwaha**

*Note: A Hindi version of this paper was presented and published at the All India Hindi Technical Conference organized by ISRO at U R Rao Satellite Centre, Bengaluru, from 5–6 February 2026.*

---

# Overview

This repository contains the implementation accompanying the research work:

**“SEOBM: A Temporal Representation for Learning-Based Orbital Behavior Modeling.”**

The project introduces **SEOBM (Self-Evolving Orbital Behavior Map)**, a structured representation designed to capture **orbital dynamics over time** in a format suitable for machine learning models.

Traditional orbital analysis typically relies on instantaneous state vectors such as:

* Cartesian position and velocity
* Orbital elements
* Short state sequences

While these representations work well for physics-based propagation and short-term prediction, they often struggle when used with learning algorithms that must reason about **temporal patterns or long-term behavior**.

SEOBM addresses this by organizing physically interpretable orbital features across a **sliding temporal memory window**, producing a fixed-size tensor representation that learning models can process effectively.

---

# Self-Evolving Orbital Behavior Map (SEOBM)

SEOBM converts orbital state vectors into a temporal tensor representation:

```
SEOBM(t) ∈ ℝ^(F × K)
```

Where:

* **F** = number of orbital behavior features
* **K** = memory window length

Each column represents a feature vector derived from the orbital state at a specific time step.

The features used in this work are:

* Orbital speed
* Radial distance
* Angular momentum magnitude
* Radial velocity component

These are computed deterministically from Cartesian position and velocity vectors.

The representation evolves using a **sliding window**, allowing models to access recent orbital history without modifying the underlying physical dynamics.

---

# Repository Structure

```
SEOBM-SAT
│
├── README.md
│
├── data/
│   └── README.md
│
└── src/
    │
    ├── ablations/
    │   └── memory_ablation.py
    │
    ├── analysis/
    │   ├── plot_learning_curves.py
    │   └── plot_memory_ablation.py
    │
    ├── envs/
    │   ├── orbital_marl_env.py
    │   └── orbital_marl_env_v2.py
    │
    ├── features/
    │   ├── behavior_features.py
    │   └── normalization.py
    │
    ├── models/
    │   ├── baseline_mlp.py
    │   ├── mappo_actor_critic.py
    │   ├── seobm_encoder.py
    │   └── seobm_lstm.py
    │
    ├── parsers/
    │   ├── parse_starlink.py
    │   └── tle_utils.py
    │
    ├── physics/
    │   ├── noise_models.py
    │   ├── propagation.py
    │   └── velocity_reconstruction.py
    │
    ├── rl/
    │   └── simple_policy_gradient.py
    │
    ├── seobm/
    │   └── seobm_tensor.py
    │
    ├── tasks/
    │
    ├── run_feature_extraction.py
    ├── run_layer1_pipeline.py
    ├── run_learning_demo.py
    ├── run_learning_demo_delayed.py
    ├── run_learning_demo_normalized.py
    ├── run_mappo_seobm_demo.py
    ├── run_mappo_seobm_train.py
    └── run_seobm_build.py
```

---

# Core Components

## Physics and Orbital Dynamics

`src/physics/`

Implements simplified orbital propagation models used for generating synthetic trajectories.

Includes:

* orbital propagation
* velocity reconstruction
* noise modeling for trajectory variability

These modules allow the project to simulate orbital motion under controlled experimental conditions.

---

## Feature Extraction

`src/features/`

Responsible for computing physically interpretable orbital behavior features.

Key modules:

**behavior_features.py**

Extracts features from orbital states:

* orbital speed
* radial distance
* angular momentum magnitude
* radial velocity

**normalization.py**

Implements feature normalization used during learning experiments.

---

## SEOBM Representation

`src/seobm/seobm_tensor.py`

Constructs the **Self-Evolving Orbital Behavior Map** by stacking feature vectors across a sliding temporal window.

This module defines the tensor construction pipeline used throughout experiments.

---

## Machine Learning Models

`src/models/`

Contains baseline and SEOBM-based learning architectures.

Includes:

**baseline_mlp.py**

Memoryless baseline model that receives only the current feature vector.

**seobm_lstm.py**

Temporal model that processes SEOBM tensors.

**seobm_encoder.py**

Feature encoding layer used for learning pipelines.

**mappo_actor_critic.py**

Actor-critic architecture used in multi-agent reinforcement learning experiments.

---

## Reinforcement Learning

`src/rl/`

Contains reinforcement learning utilities used in the MARL demonstrations.

`simple_policy_gradient.py` provides a lightweight implementation for policy optimization.

---

## Multi-Agent Environments

`src/envs/`

Defines synthetic orbital multi-agent environments used for reinforcement learning demonstrations.

Examples include:

* simplified orbital interaction environments
* coordination reward structures for agents

These environments are intentionally simplified and are used only for representation experiments.

---

## Experiment Runners

The repository provides several runnable experiment pipelines located directly in `src/`.

### Feature Extraction

```
python src/run_feature_extraction.py
```

Computes orbital behavior features from trajectory data.

---

### SEOBM Construction

```
python src/run_seobm_build.py
```

Builds SEOBM tensors from extracted feature sequences.

---

### Synthetic Trajectory Pipeline

```
python src/run_layer1_pipeline.py
```

Generates synthetic orbital trajectories used in experiments.

---

### Supervised Learning Demonstrations

Short-horizon prediction:

```
python src/run_learning_demo.py
```

Delayed prediction task:

```
python src/run_learning_demo_delayed.py
```

Normalized learning experiment:

```
python src/run_learning_demo_normalized.py
```

---

### Multi-Agent Reinforcement Learning

MAPPO demonstration:

```
python src/run_mappo_seobm_demo.py
```

MAPPO training pipeline:

```
python src/run_mappo_seobm_train.py
```

---

# Ablation Studies

`src/ablations/memory_ablation.py`

This experiment evaluates the impact of **memory window size K** on prediction performance.

It demonstrates how increasing temporal context improves learning in delayed prediction tasks.

---

# Analysis and Visualization

`src/analysis/`

Includes scripts used to generate plots and analyze training results.

Examples:

```
python src/analysis/plot_learning_curves.py
python src/analysis/plot_memory_ablation.py
```

---

# Dataset Policy

The repository **does not contain the datasets used during experiments.**

This is intentional and follows the methodology described in the research paper.

The experimental setup uses a **two-layer data strategy**:

### Layer-2: Real Satellite Data

Publicly available Starlink satellite data is used only to define **realistic orbital regimes** such as altitude and inclination ranges.

This data is **not used for training or evaluation.**

---

### Layer-1: Synthetic Orbital Trajectories

All experiments are performed using **synthetic trajectories generated through simplified orbital propagation models.**

Synthetic data allows:

* reproducible experiments
* full control of ground truth
* isolation of representation effects

---

# Why the `data/` Folder Is Empty

The `data/` directory exists only as a placeholder.

Large datasets and generated trajectories are not stored in the repository because:

* Git repositories are not designed to store large datasets
* experiments rely on synthetic trajectory generation
* datasets can be reproduced using the provided pipelines

---

# Scope of This Work

This project focuses on **representation learning for orbital behavior modeling.**

The repository **does not propose**:

* operational SSA systems
* collision avoidance algorithms
* satellite control policies
* real-world safety guarantees

The experiments are conducted entirely in **controlled synthetic environments** and should be interpreted accordingly.

---

# Reproducibility

All experiments can be reproduced using the code provided in this repository.

The design intentionally separates:

* orbital physics
* feature representation
* learning pipelines

This allows researchers to isolate the effects of temporal representation when studying orbital behavior.

---

# Future Work

Possible extensions include:

* richer orbital feature sets
* higher-fidelity propagation models
* uncertainty-aware learning methods
* integration with real observational datasets

SEOBM provides a foundation for further research at the intersection of **astrodynamics and machine learning**.
