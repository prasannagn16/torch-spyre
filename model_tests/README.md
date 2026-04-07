# pytorch_collaboration

This repository contains code and resources for various tracks under the PyTorch collaboration effort.

## Overview

The repository is organized into the following key tracks:

1. **Evaluation Framework**
2. **Model-Centric Op-Level Test Cases**
3. **Ops Discovery / Enablement**

---

## Evaluation Framework

A framework designed to evaluate models across multiple metrics once they have been enabled.  
It provides a standardized way to validate model performance and correctness.

---

## Model-Centric Op-Level Test Cases

This module contains operation-level test cases tailored to specific models.  
These tests help ensure correctness and stability of individual operations within each model.

---

## Ops Discovery / Enablement

This track includes tools and scripts to identify and enable operations required for different models.

### Key components:
- **Model-specific scripts**  
  Used to analyze and identify the set of operations required by each model.

- **`fallbacks.py` (per model)**  
  Maintains fallback implementations for unsupported operations.  
  Since multiple developers may contribute:
  - Please raise a PR for any changes  
  - Ensure all updates are merged into the `main` branch

- **Sample BERT script**  
  A reference script to identify operations for any model.  
  You can run:
  - The Jupyter notebook version, or  
  - The standalone `.py` script

---

## Contribution Guidelines

- Follow existing structure and naming conventions  
- Submit changes via Pull Requests  
- Coordinate updates to shared files like `fallbacks.py` to avoid conflicts  

---
