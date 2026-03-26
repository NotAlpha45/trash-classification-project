# Multimodal Trash Classification: A Two-Phase Study (Supervised Fusion and Retrieval-Augmented Classification)

This repository presents a two-phase research workflow for four-class trash-bin classification (`Black`, `Blue`, `Green`, `TTR`) using image and text signals.

- Phase 1 establishes supervised baselines: image-only, text-only, and multimodal fusion.
- Phase 2 studies retrieval-augmented classification (RAC) using multimodal memory in ChromaDB, including robustness under class imbalance and memory growth.
- A central objective is to evaluate multiple **vector heuristics** for retrieval scoring and aggregation (implemented as RAC variants).

The project is designed as an empirical pipeline: model building, retrieval design, controlled ablations, stress testing, and consolidated reporting.

## Table of Contents

- [Abstract](#abstract)
- [1. Research Context](#1-research-context)
- [Contributions](#contributions)
- [2. Dataset and Task](#2-dataset-and-task)
- [3. Phase 1: Supervised Multimodal Baselines](#3-phase-1-supervised-multimodal-baselines)
- [4. Phase 2: Retrieval-Augmented Classification](#4-phase-2-retrieval-augmented-classification)
   - [4.3 Vector Heuristic Formulations](#43-vector-heuristic-formulations)
   - [4.4 Phase 2 Experimental Protocols](#44-phase-2-experimental-protocols)
- [5. Installation](#5-installation)
- [6. Repository Structure](#6-repository-structure)
- [7. Reproducibility Workflow](#7-reproducibility-workflow)
- [8. Empirical Results](#8-empirical-results)
  - [8.1 Phase 1 Results](#81-phase-1-results)
  - [8.2 Phase 2 Results](#82-phase-2-results)
   - [8.3 Interpretation Relative to Vector-Heuristic Evaluation](#83-interpretation-relative-to-vector-heuristic-evaluation)
- [9. Main Artifacts](#9-main-artifacts)
- [10. Technology Stack](#10-technology-stack)
- [11. Limitations and Scope](#11-limitations-and-scope)
- [12. Acknowledgments](#12-acknowledgments)
- [13. License](#13-license)

## Abstract

This repository presents a two-phase study of multimodal trash-bin classification. Phase 1 establishes supervised baselines (MobileNetV3 image model, DistilBERT text model, and weighted late fusion). Phase 2 evaluates retrieval-augmented classification as a vector-heuristic problem, where class decisions are produced from embedding-neighborhood evidence under multiple scoring rules. Beyond standard evaluation, we report controlled experiments on density normalization, minority-class imbalance, and continual-memory growth. The results indicate that density-aware heuristics are consistently more robust than unnormalized vote/distance aggregation, especially under severe imbalance and constrained retrieval memory.

## 1. Research Context

This work investigates two complementary questions:

1. How strong is multimodal supervised classification when image and text are fused at logit level?
2. Can retrieval-augmented decision rules improve robustness and minority-class behavior under realistic memory constraints?
3. Which vector heuristics (RAC scoring variants) produce the most reliable performance across standard, imbalanced, and continual-memory settings?

The repository therefore separates model-centric learning (Phase 1) from retrieval-centric reasoning and stress tests (Phase 2).

## Contributions

1. A complete two-phase multimodal pipeline spanning supervised fusion and retrieval-augmented inference.
2. A unified implementation and comparison of vector heuristics (`majority_vote`, `idw`, `global_dnds`, `local_dnds`, `kde_dnds`, and `traditional` reference).
3. A report-style evaluation suite including fixed/tuned protocols, database-level imbalance simulation, and continual-memory analysis.
4. Reproducible experiment artifacts (JSON/CSV/figures) and notebook workflows for end-to-end regeneration.

## 2. Dataset and Task

**Task:** 4-way classification of trash-bin class labels.

**Classes:**
- `Black`
- `Blue`
- `Green`
- `TTR`

**Image splits:**
- `dataset/CVPR_2024_dataset_Train/`
- `dataset/CVPR_2024_dataset_Val/`
- `dataset/CVPR_2024_dataset_Test/`

**Text splits:**
- `dataset_text/train.csv`
- `dataset_text/val.csv`
- `dataset_text/test.csv`

In this project, text features are derived from filename-like textual inputs and used as an additional modality.

## 3. Phase 1: Supervised Multimodal Baselines

Phase 1 establishes supervised reference systems.

### Models

- **Image model:** MobileNetV3-Large (transfer learning)
- **Text model:** DistilBERT-base-uncased
- **Fusion model:** weighted logit fusion

Fusion equation:

$$
W_{fused} = \alpha W_{text} + (1-\alpha)W_{image}
$$

### Phase 1 notebooks

1. `notebooks/phase1/image_model_experiment_mobilenetv3.ipynb`
2. `notebooks/phase1/distilbert_text_model.ipynb`
3. `notebooks/phase1/multimodal_fusion.ipynb`

These notebooks train/evaluate unimodal systems and quantify fusion gains.

## 4. Phase 2: Retrieval-Augmented Classification

Phase 2 introduces RAC over a multimodal retrieval database.

In this project, the RAC techniques are treated as **vector heuristics**: algorithmic rules that convert retrieved embedding neighborhoods into final class decisions.

### 4.1 RAC overview

- Build persistent image/text collections in ChromaDB.
- Retrieve nearest neighbors for both modalities.
- Aggregate evidence via multiple scoring rules.

Implemented scoring variants:

- `majority_vote`
- `idw`
- `global_dnds`
- `local_dnds`
- `kde_dnds`
- `traditional` (reference baseline)

### 4.2 Default experiment configuration

From `src/phase2/config.py`:

- `k_vote = 10`
- `K_density = 50`
- `K_density_sweep = [10, 25, 50, 75, 100]`
- `alpha = 0.5`
- `alpha_sweep = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]`
- `imbalance_ratios = [2, 3, 5, 10]`
- `kde_bandwidth = 0.5`
- `kde_bandwidth_sweep = [0.1, 0.25, 0.5, 1.0]`

### 4.3 Vector Heuristic Formulations

Let $\mathcal{N}^{(m)}_k(q)$ denote the top-$k$ neighbors retrieved for query $q$ in modality $m \in \{I, T\}$ (image/text), with distances $d_i^{(m)}$, and let $\epsilon > 0$ be a small stabilizer.

Notation summary:

| Symbol | Meaning |
|---|---|
| $q$ | query sample |
| $m \in \{I, T\}$ | modality (image or text) |
| $\mathcal{N}^{(m)}_k(q)$ | top-$k$ neighbors of $q$ in modality $m$ |
| $d_i^{(m)}$ | retrieval distance of neighbor $i$ |
| $y_i$ | class label of neighbor $i$ |
| $k_{vote}$ | voting/IDW neighborhood size |
| $K_{density}$ | local density neighborhood size |
| $\rho_m(c)$ | global class density in modality $m$ |
| $\rho_m^{local}(c)$ | local class density around query |
| $\tilde{\rho}_m(c)$ | KDE-smoothed class density |
| $h$ | KDE bandwidth |
| $\alpha$ | image-text fusion weight |
| $\epsilon$ | numerical stability constant |

Phase 2 RAC fusion uses:

$$
S(c) = \alpha S_I(c) + (1-\alpha)S_T(c), \quad \hat{y} = \arg\max_c S(c)
$$

where $S_I(c)$ and $S_T(c)$ are class scores derived from image and text retrieval, respectively.

1. Majority Vote

$$
S_m(c) = \sum_{i \in \mathcal{N}^{(m)}_{k_{vote}}(q)} \mathbf{1}[y_i = c]
$$

2. Inverse Distance Weighting (IDW)

$$
S_m(c) = \sum_{i \in \mathcal{N}^{(m)}_{k_{vote}}(q),\, y_i=c} \frac{1}{d_i^{(m)} + \epsilon}
$$

3. Global DNDS

Using global class density $\rho_m(c)$ from database counts:

$$
\rho_m(c) = \frac{N_m(c)}{\sum_{c'} N_m(c')}, \quad
S_m^{global}(c) = \frac{\sum_{i \in \mathcal{N}^{(m)}_{k_{vote}}(q),\, y_i=c} \frac{1}{d_i^{(m)} + \epsilon}}{\rho_m(c)}
$$

4. Local DNDS

Using local neighborhood density from top-$K_{density}$ neighbors:

$$
\rho_m^{local}(c) = \frac{\|\{i \in \mathcal{N}^{(m)}_{K_{density}}(q): y_i=c\}\|}{K_{density}}, \quad
S_m^{local}(c) = \frac{\sum_{i \in \mathcal{N}^{(m)}_{k_{vote}}(q),\, y_i=c} \frac{1}{d_i^{(m)} + \epsilon}}{\rho_m^{local}(c)}
$$

5. KDE-DNDS

Replacing local frequency with Gaussian KDE class density (bandwidth $h$):

$$
   \tilde{\rho}_m(c) = \sum_{j: y_j=c} \exp\left(-\frac{(d_j^{(m)})^2}{2h^2}\right), \quad
S_m^{kde}(c) = \frac{\sum_{i \in \mathcal{N}^{(m)}_{k_{vote}}(q),\, y_i=c} \frac{1}{d_i^{(m)} + \epsilon}}{\tilde{\rho}_m(c)}
$$

6. Traditional reference

Phase 1 model logits are combined as a non-retrieval baseline and evaluated under the same Phase 2 test protocol.

### 4.4 Phase 2 Experimental Protocols

The Phase 2 study includes four experiments.

1. Standard RAC evaluation (`02_evaluation.ipynb`)
- Evaluate all heuristics on a fixed test split.
- Sweep/tune `alpha`, `K_density`, and KDE bandwidth.
- Report accuracy, macro/weighted F1, per-class precision/recall/F1, confusion matrix, and latency.

2. Fixed-alpha evaluation (`02.1_evaluation.ipynb`)
- Repeat evaluation with alpha tuning disabled.
- Assess whether heuristic ranking remains stable under fixed fusion weighting.

3. Imbalance stress test (`03_imbalance_experiment.ipynb`)
- Build ratio-specific imbalanced DBs by retaining majority classes and subsampling minority classes.
- For ratio $r$, minority target per class is:

$$
n_{minority}(c) = \max\left(1, \left\lfloor \frac{\min_{c' \in \mathcal{C}_{maj}} N(c')}{r} \right\rfloor\right)
$$

- Evaluate heuristics at $r \in \{2,3,5,10\}$ and quantify minority robustness.

4. Continual-memory experiment (`04_continual_learning.ipynb`)
- Evaluate performance as retrieval memory grows from 10% to 100%.
- For each fraction $p \in \{0.1,0.2,\dots,1.0\}$, construct the corresponding DB subset and re-evaluate heuristics.
- Analyze trend behavior (improvement, saturation, stability) as memory increases.

### 4.5 Phase 2 notebooks

1. `notebooks/phase2/01_db_population.ipynb`
   - Populates `chroma_db/` collections from train/val data.
2. `notebooks/phase2/02_evaluation.ipynb`
   - RAC evaluation with alpha tuning.
3. `notebooks/phase2/02.1_evaluation.ipynb`
   - RAC evaluation with fixed alpha (no sweep).
4. `notebooks/phase2/03_imbalance_experiment.ipynb`
   - Database-level minority subsampling at controlled ratios.
5. `notebooks/phase2/04_continual_learning.ipynb`
   - Performance as retrieval memory grows.
6. `notebooks/phase2/05_results_summary.ipynb`
   - Regenerates final figures/tables from saved artifacts.

## 5. Installation

### Prerequisites

- Python 3.12+
- `uv` package manager
- CUDA-capable GPU (optional but recommended)

### Setup

1. Clone repository

```bash
git clone https://github.com/NotAlpha45/trash-classification-project.git
cd trash-classification-project
```

2. Install dependencies

```bash
uv sync
```

3. Verify runtime

```bash
uv run python -c "import torch; print(torch.__version__); print(torch.cuda.is_available())"
```

## 6. Repository Structure

```text
trash-classification-project/
├── notebooks/
│   ├── phase1/
│   │   ├── image_model_experiment_mobilenetv3.ipynb
│   │   ├── distilbert_text_model.ipynb
│   │   └── multimodal_fusion.ipynb
│   └── phase2/
│       ├── 01_db_population.ipynb
│       ├── 02_evaluation.ipynb
│       ├── 02.1_evaluation.ipynb
│       ├── 03_imbalance_experiment.ipynb
│       ├── 04_continual_learning.ipynb
│       └── 05_results_summary.ipynb
├── src/phase2/
│   ├── config.py
│   ├── data_utils.py
│   ├── db_client.py
│   ├── embedders.py
│   ├── evaluation.py
│   ├── gpu_utils.py
│   ├── imbalance.py
│   ├── scoring.py
│   ├── traditional.py
│   └── visualization.py
├── dataset/
├── dataset_text/
├── models/
├── text_models/
├── chroma_db/
├── chroma_db_continual/
├── results/phase2/
├── figures/phase2/
├── pyproject.toml
└── README.md
```

## 7. Reproducibility Workflow

For full regeneration of Phase 2 artifacts, run notebooks in this order:

1. `notebooks/phase2/01_db_population.ipynb`
2. `notebooks/phase2/02_evaluation.ipynb` (or `02.1_evaluation.ipynb` for fixed-alpha protocol)
3. `notebooks/phase2/03_imbalance_experiment.ipynb`
4. `notebooks/phase2/04_continual_learning.ipynb`
5. `notebooks/phase2/05_results_summary.ipynb`

Launch Jupyter:

```bash
uv run jupyter notebook
```

## 8. Empirical Results

### 8.1 Phase 1 Results

| Model | Accuracy | Macro F1 | Weighted F1 | Inference Time |
|---|---:|---:|---:|---:|
| Image Only (MobileNetV3) | 72.16% | 0.7150 | 0.7208 | 0.27 ms/sample |
| Text Only (DistilBERT) | 77.23% | 0.7711 | 0.7735 | 1.15 ms/sample |
| Multimodal Fusion (alpha = 0.60) | **82.16%** | **0.8177** | **0.8212** | 1.43 ms/sample |

### 8.2 Phase 2 Results

From `results/phase2/final_results_summary.csv` (fixed-alpha setting):

| Variant | Accuracy | Macro F1 | TTR F1 |
|---|---:|---:|---:|
| majority_vote | 82.94% | 0.8236 | 0.8225 |
| idw | 83.72% | 0.8319 | 0.8442 |
| global_dnds | **85.25%** | **0.8481** | 0.8706 |
| local_dnds | 82.18% | 0.8176 | 0.8345 |
| kde_dnds | 85.23% | 0.8480 | **0.8716** |
| traditional | 81.84% | 0.8137 | 0.8130 |

#### Imbalance robustness (ratio 10:1)

From `results/phase2/imbalance_summary.csv`:

- `majority_vote` TTR F1: 0.5843
- `idw` TTR F1: 0.6201
- `global_dnds` TTR F1: 0.7651
- `local_dnds` TTR F1: 0.6535
- `kde_dnds` TTR F1: 0.7669

Observation: density-aware global/KDE variants retain substantially stronger minority performance under severe scarcity.

#### Continual-learning trend

From `results/phase2/continual_learning_results.json`:

- Example (`global_dnds`): macro F1 improves from 0.7679 (10% memory) to 0.8481 (100% memory).

This supports the expected benefit of retrieval memory growth on decision quality.

### 8.3 Interpretation Relative to Vector-Heuristic Evaluation

- In full-memory evaluation, density-normalized heuristics (`global_dnds`, `kde_dnds`) deliver the best aggregate metrics.
- In severe imbalance settings, density-aware heuristics preserve minority performance substantially better than vote-only or plain IDW rules.
- In continual-memory tests, macro F1 rises as memory size increases, indicating that heuristic reliability depends on richer neighborhood evidence.

Overall, the experiments support the core research objective: vector heuristics with explicit density normalization are more robust under shift and scarcity than unnormalized neighborhood aggregation.

## 9. Main Artifacts

### Phase 1 artifacts

- `models/mobilenetv3_best.pth`
- `text_models/distilbert_best.pth`
- model-specific visualizations under `models/`, `text_models/`, and `figures/`

### Phase 2 artifacts

- `results/phase2/evaluation_results.json`
- `results/phase2/evaluation_no_alpha_results.json`
- `results/phase2/imbalance_results.json`
- `results/phase2/continual_learning_results.json`
- `results/phase2/final_results_summary.csv`
- `figures/phase2/*.png`

## 10. Technology Stack

- PyTorch / Torchvision
- HuggingFace Transformers
- ChromaDB
- NumPy / pandas / scikit-learn
- Matplotlib / Seaborn
- Jupyter Notebooks

## 11. Limitations and Scope

- Results reflect the current dataset and class taxonomy only.
- Text modality is filename-based and may encode dataset-specific naming bias.
- Retrieval sensitivity to embedding/checkpoint changes should be expected.
- Further external validation is needed before deployment-level claims.

## 12. Acknowledgments

- MobileNetV3: *Searching for MobileNetV3* (Howard et al.)
- DistilBERT: *DistilBERT, a distilled version of BERT* (Sanh et al.)
- CVPR 2024 trash classification dataset
- University of Calgary, ENSF617 (Winter 2026)

## 13. License

This repository is maintained for educational and coursework use.
