# CoMODAL: Coalition-based Collaboration for On-demand Bi-modal Automated Delivery

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)

## 🚀 Overview

**CoMODAL** is a routing and task-allocation framework for cooperative bi-modal urban last-mile logistics. The repository provides the learned policy, classical comparators, and coalition analysis that accompany *Coalition-based collaboration for on-demand bi-modal automated delivery* by Farzan Moosavi and Bilal Farooq.

Traditional automated delivery systems operate in silos. CoMODAL coordinates Unmanned Aerial Vehicles (UAVs) and Autonomous Delivery Robots (ADRs) within shared coalitions, so that firms can overcome single-mode payload, battery, and spatial constraints while meeting pickup-and-delivery, time-window, electric-energy, and mandatory-service requirements.

The same framework extends naturally to multi-modal operations. The fleet model, heterogeneous attention encoder, and coalition engine can accommodate additional agent types such as Autonomous Ground Vehicles (AGVs), alternative drone platforms, or human-operated couriers by registering new mode profiles and allowed-mode sets in the dispatch and benchmark configurations.

## ✨ Key Features

* **HetGAT route policy** trained with REINFORCE leave-one-out for heterogeneous UAV + ADR dispatch.
* **Guarded commitment overlay** for replay-assisted decoding at scale.
* **Strong ALNS and MIP comparators** (Gurobi and OR-Tools SCIP).
* **Learned baselines** adapted from Kool et al. 2019 (Attention Model) and related architectures.
* **Coalition analysis** (Shapley, equal-profit, and proportional-allocation methods) on matched order-ownership sets.

## 🏗️ System Architecture

1. **Request Management:** Rolling-horizon dispatcher ingests on-demand delivery tasks under pickup-and-delivery, time-window, and battery constraints.
2. **Coalition Engine:** Evaluates matched order-ownership sets and computes Shapley, equal-profit, and proportional cost allocations.
3. **Bi-modal Router:** Generates energy-aware UAV and ADR routes via the HetGAT policy, the guarded-commitment overlay, or the strong ALNS and MIP comparators.
4. **Reproduction Harness:** Loads checkpoints and evaluation books from the release protocol, verifies SHA-256 digests, and replays matched paired-benchmark cells.

## 🛠️ Getting Started

### Prerequisites

Python 3.12. Install dependencies from the repository root.

```bash
python -m pip install -r requirements.txt
```

The exact-solver modules use optional dependencies `ortools` and `gurobipy`.

### Reproduce

```bash
python src/reproduce.py --requests 25 --pack cap50 --method guarded --book 0 --output run.csv
```

Runs a single matched book from a checkpoint. Supply multiple `--book` indices for a paired subset. Checkpoint and bank SHA-256 digests are recorded in `data/routing_protocol.json`.

## 📁 Contents

* `src/` — reproduction source, including the HetGAT policy, dispatcher, guarded commitment overlay, classical comparators, and learned baselines.
* `data/` — evaluation books and the release protocol with per-checkpoint digests.
* `models/` — trained policy checkpoints for the four demand stages.

## 🤖 AI Use

Claude and ChatGPT were used to develop this code.

## 📘 Citation

```bibtex
@article{moosavi2026coalition,
  title   = {Coalition-based collaboration for on-demand bi-modal automated delivery},
  author  = {Moosavi, Farzan and Farooq, Bilal},
  journal = {Transportation Research Part E},
  year    = {2026}
}
```
