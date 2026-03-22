# Phase 2 Coding Agent Prompt — RAC Trash Classification

## Context

You are implementing **Phase 2** of a multimodal trash classification research project. Phase 1 already exists at:
**https://github.com/NotAlpha45/trash-classification-project**

Phase 1 implements a traditional supervised classifier using:
- **Image model**: MobileNetV3-Large (pretrained, fine-tuned with a 3-layer classifier head)
- **Text model**: DistilBERT-base-uncased (fine-tuned on image filenames)
- **Fusion**: Weighted logit-level fusion (`W_fused = α·W_text + (1-α)·W_image`)
- **Dataset**: CVPR 2024 Trash Classification dataset with 4 Calgary waste bin classes: `Black`, `Blue`, `Green`, `TTR`
- **Toolchain**: Python 3.12, PyTorch 2.10+, HuggingFace Transformers, `uv` package manager

The existing `pyproject.toml` dependencies are:
```
torch>=2.10.0, torchvision>=0.25.0, transformers, scikit-learn, matplotlib, seaborn, ipykernel, markupsafe
```

The dataset structure is:
```
dataset/
  CVPR_2024_dataset_Train/   # Training images, organized in subfolders by class
  CVPR_2024_dataset_Val/     # Validation images
  CVPR_2024_dataset_Test/    # Test images
dataset_text/
  train.csv                  # columns: text (filename), label (class)
  val.csv
  test.csv
```

---

## Your Task

Implement **Phase 2: Retrieval-Augmented Classification (RAC)** as a set of organized Jupyter notebooks under a new `notebooks/phase2/` folder. Each notebook has a single responsibility and they are designed to be run sequentially. Intermediate results (predictions, scores, metrics) are saved to disk as `.pkl` or `.csv` files under `results/phase2/` so later notebooks can load them without re-running expensive steps.

### Folder & File Structure

```
notebooks/
├── phase1/                          ← move existing Phase 1 notebooks here
│   ├── image_model_experiment_mobilenetv3.ipynb
│   ├── distilbert_text_model.ipynb
│   └── multimodal_fusion.ipynb
└── phase2/
    ├── 01_db_population.ipynb       ← populate ChromaDB from training set (run once)
    ├── 02_scoring_variants.ipynb    ← implement & evaluate all 5 scoring functions on test set
    ├── 03_imbalance_experiment.ipynb← controlled imbalance ratio sweep
    ├── 04_continual_learning.ipynb  ← progressive DB growth simulation
    └── 05_results_visualization.ipynb ← load all saved results, generate all final figures

results/
└── phase2/
    ├── scoring_variants_predictions.pkl   ← saved by 02
    ├── imbalance_results.pkl              ← saved by 03
    ├── continual_learning_results.pkl     ← saved by 04
    └── metrics_summary.csv               ← saved by 02, used by 05

figures/
└── phase2/
    ├── scoring_comparison.png
    ├── minority_f1_vs_imbalance.png
    ├── alpha_sensitivity.png
    ├── continual_learning_curve.png
    ├── confusion_matrices_phase2.png
    └── phase2_vs_phase1.png

chroma_db/                           ← persistent ChromaDB storage (gitignored)
```

Add `chroma_db/` and `results/phase2/` to `.gitignore`.

### Per-Notebook Responsibilities

**`01_db_population.ipynb`**
- Initialize persistent ChromaDB client at `./chroma_db`
- Create `trash_image_db` and `trash_text_db` collections
- Populate both collections from the training set in batches
- At the end, print collection counts to confirm population succeeded
- Include an idempotency check: skip population if `collection.count() == len(train_dataset)`
- This notebook is run **once**. All others assume the DB is already populated.

**`02_scoring_variants.ipynb`**
- Load the test set
- Implement all 5 scoring variant functions (majority_vote, idw, global_dnds, local_dnds, traditional)
- Run all variants on the full test set (no imbalance simulation here)
- Run alpha sensitivity sweep for `local_dnds`
- Save predictions and metrics to `results/phase2/scoring_variants_predictions.pkl` and `metrics_summary.csv`
- Print the final summary results table

**`03_imbalance_experiment.ipynb`**
- Load test set
- Implement imbalance simulation via DB result post-filtering
- Run all 5 variants at ratios `[10, 50, 100]`
- Focus metric: per-class F1, especially TTR and Green
- Save results to `results/phase2/imbalance_results.pkl`

**`04_continual_learning.ipynb`**
- Simulate progressive DB growth (10% → 100% of training set, in 10% steps)
- At each step, evaluate `local_dnds` on the full test set
- Save results to `results/phase2/continual_learning_results.pkl`

**`05_results_visualization.ipynb`**
- Load all `.pkl` and `.csv` files from `results/phase2/`
- Generate and save all 6 figures to `figures/phase2/`
- This notebook produces no new computations — purely visualization

Each notebook must begin with a Markdown cell stating its purpose, its inputs (what files it expects), and its outputs (what files it produces).

---

## Architecture Overview

Instead of a trained classification head, Phase 2 uses two ChromaDB vector databases populated with embeddings of the training set, and classifies new items by retrieving the most similar neighbors and computing a density-normalized distance score.

### Two Collections in ChromaDB

**Collection 1: `trash_image_db`**
- Stores image embeddings of all training samples
- Each record metadata: `{"label": "Black"|"Blue"|"Green"|"TTR", "filename": "..."}`

**Collection 2: `trash_text_db`**
- Stores text embeddings of all training filenames (same data as Phase 1 text model)
- Each record metadata: `{"label": "Black"|"Blue"|"Green"|"TTR", "filename": "..."}`

### Embedding Models

**For images — use `openclip-ViT-B-32`** (via the `open_clip` library):
- This is a strong open-source CLIP variant that produces 512-dim embeddings
- It is natively supported by ChromaDB via `OpenCLIPEmbeddingFunction`
- Use it to encode images to vectors; store the vectors in `trash_image_db`

```python
from chromadb.utils.embedding_functions import OpenCLIPEmbeddingFunction
image_ef = OpenCLIPEmbeddingFunction()  # uses ViT-B-32 by default
```

**For text — use `sentence-transformers/all-MiniLM-L6-v2`** (via `sentence-transformers`):
- 22M parameter model, 384-dim embeddings, fast and ChromaDB-compatible
- Natively supported by ChromaDB via `SentenceTransformerEmbeddingFunction`

```python
from chromadb.utils.embedding_functions import SentenceTransformerEmbeddingFunction
text_ef = SentenceTransformerEmbeddingFunction(model_name="all-MiniLM-L6-v2")
```

### ChromaDB Setup

Use a persistent local client so the DB survives across notebook runs:

```python
import chromadb
client = chromadb.PersistentClient(path="./chroma_db")

image_collection = client.get_or_create_collection(
    name="trash_image_db",
    embedding_function=image_ef,
    metadata={"description": "Image embeddings for RAC experiment"}
)

text_collection = client.get_or_create_collection(
    name="trash_text_db",
    embedding_function=text_ef,
    metadata={"description": "Text (filename) embeddings for RAC experiment"}
)
```

---

## The Scoring Function — DNDS (Density-Normalized Distance Score)

This is the core novel contribution. Do NOT use simple majority voting or flat inverse distance weighting. Implement the following exactly.

### Step 1: Query both collections

For a given query, retrieve the top-**K** neighbors from each collection (use K=50 for density estimation):

```python
image_results = image_collection.query(
    query_images=[query_image_array],  # numpy array
    n_results=K,
    include=["metadatas", "distances"]
)

text_results = text_collection.query(
    query_texts=[query_filename],
    n_results=K,
    include=["metadatas", "distances"]
)
```

Note: ChromaDB returns cosine **distance** (not similarity), so distance ∈ [0, 2]. Closer = smaller distance.

### Step 2: Compute DNDS score for each modality

Given the K neighbors for modality `m` ∈ {image, text}:

```
Classes = {Black, Blue, Green, TTR}
ε = 1e-6  (prevents division by zero)

For each class c:
    # Local density: fraction of K neighbors belonging to class c
    N_c_local = count of neighbors in top-K where label == c
    ρ_c = N_c_local / K

    # Inverse-distance weighted sum over the smaller k neighbors (k=10) of class c
    neighbors_c = [i for i in top_k_neighbors if label_i == c]  # top k=10, not K=50
    raw_score = sum(1 / (d_i + ε) for i in neighbors_c)

    # Density-normalized score
    if ρ_c == 0:
        S_m(c) = 0
    else:
        S_m(c) = raw_score / ρ_c
```

**Important**: Use `K=50` for density estimation and `k=10` for the inner vote sum. The top-K is the density window; the top-k is the classification neighborhood. Both come from the same query (just use the first k of the K results for scoring).

### Step 3: Fuse image and text scores

```
S_final(c) = α * S_image(c) + (1 - α) * S_text(c)
ŷ = argmax_c S_final(c)
```

Use `α = 0.5` as the default. Run a sensitivity sweep over `α ∈ [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0]` and report the best.

---

## Experiment Design

### Section 1 — Database Population

Populate both ChromaDB collections with all training set samples. Add records in batches of 100 to avoid memory issues. Each record:

```python
image_collection.add(
    ids=[f"img_{idx}"],
    images=[image_numpy_array],       # shape: (H, W, 3), dtype: uint8
    metadatas=[{"label": label, "filename": filename}]
)

text_collection.add(
    ids=[f"txt_{idx}"],
    documents=[filename_string],
    metadatas=[{"label": label, "filename": filename}]
)
```

Add a check: if `image_collection.count() == len(train_dataset)`, skip re-population (idempotent).

### Section 2 — Baseline Comparisons

Implement all 5 scoring variants as separate functions so they can be swapped in easily:

| Variant | Description |
|---|---|
| `majority_vote` | Top-k neighbors, class with most votes wins |
| `idw` | Inverse distance weighting, no imbalance correction |
| `global_dnds` | Density correction using global class counts in DB (`N_c / N`) |
| `local_dnds` | **Our method** — density correction using local K-neighborhood |
| `traditional` | Load Phase 1 fusion model checkpoints, run inference (for comparison) |

For the traditional baseline, load the saved checkpoints from Phase 1:
```python
image_checkpoint = torch.load("models/mobilenetv3_best.pth")
text_checkpoint = torch.load("text_models/distilbert_best.pth")
```

### Section 3 — Imbalance Simulation

For the imbalance experiment, **do not retrain or rebuild the full DB**. Instead, simulate imbalance at query time by sub-sampling the DB results:

- For a given imbalance ratio `r:1` (majority:minority), limit the maximum neighbors returned per majority class to `r * (K / num_minority_classes)` by post-filtering the ChromaDB results before scoring.
- Run all 5 variants at ratios: `[10, 50, 100]`
- The minority classes are: `TTR` and `Green` (lower frequency in real Calgary waste)
- The majority classes are: `Black` and `Blue`

### Section 4 — Continual Learning Curve

Simulate progressive database growth:
- Start with 10% of training data in the DB
- Incrementally add 10% at a time (10%, 20%, ..., 100%)
- At each step, evaluate `local_dnds` on the full test set
- Plot: x-axis = DB size (%), y-axis = macro F1

This directly demonstrates the "every new instance empowers the system" claim.

---

## Evaluation Metrics

Compute for every variant and every imbalance ratio:

```python
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    classification_report,
    confusion_matrix
)

metrics = {
    "accuracy": accuracy_score(y_true, y_pred),
    "macro_f1": f1_score(y_true, y_pred, average="macro"),
    "weighted_f1": f1_score(y_true, y_pred, average="weighted"),
    "per_class_f1": f1_score(y_true, y_pred, average=None, labels=CLASS_NAMES),
    "report": classification_report(y_true, y_pred, target_names=CLASS_NAMES)
}
```

Also record inference latency (ms/sample) for each variant using `time.perf_counter()`.

---

## Visualizations

Generate and save all plots to `figures/phase2/`:

1. **`scoring_comparison.png`** — Bar chart: Macro F1 for all 5 variants (no imbalance)
2. **`minority_f1_vs_imbalance.png`** — Line chart: TTR F1 and Green F1 vs. imbalance ratio, for all 5 variants (the headline result)
3. **`alpha_sensitivity.png`** — Line chart: Macro F1 vs. α for `local_dnds`
4. **`continual_learning_curve.png`** — Line chart: Macro F1 vs. DB size %
5. **`confusion_matrices_phase2.png`** — 2×2 grid of confusion matrices: majority_vote | idw | global_dnds | local_dnds
6. **`phase2_vs_phase1.png`** — Side-by-side bar: Phase 1 fusion (82.16% baseline) vs. best RAC variant

---

## Dependencies to Add

This project uses **`uv`** as the Python package and project manager. Do NOT use `pip` directly. All dependency management must go through `uv`.

Add the new Phase 2 dependencies:

```bash
uv add chromadb open-clip-torch sentence-transformers pillow pandas tqdm
```

Verify compatibility: `chromadb>=0.6.0`, `open-clip-torch>=2.24.0`, `sentence-transformers>=3.0.0`

After adding, `uv` will automatically update both `pyproject.toml` and `uv.lock`. Commit both files.

### Running Notebooks

Always launch Jupyter via `uv` to ensure the correct virtual environment is used:

```bash
uv run jupyter notebook
# or
uv run jupyter lab
```

Do NOT activate a virtualenv manually or use a system Python. The `.python-version` file in the repo root pins the interpreter; `uv` respects this automatically.

### Running Scripts

Any standalone Python scripts (e.g. for batch DB population outside of a notebook) must also be run via:

```bash
uv run python src/phase2/some_script.py
```

### Making `src/phase2/` Importable

Since the notebooks live at `notebooks/phase2/` and the source modules live at `src/phase2/`, the notebooks need `src/` on the Python path. Add this at the top of every Phase 2 notebook before any imports:

```python
import sys
from pathlib import Path

# Make src/ importable regardless of where the notebook is run from
sys.path.insert(0, str(Path("../..").resolve()))  # from notebooks/phase2/ → repo root
```

Alternatively, register the package with uv in editable mode by ensuring `pyproject.toml` includes:

```toml
[tool.uv]
package = true
```

And that `src/phase2/__init__.py` exists (even if empty). This lets `uv run` resolve `from phase2.scoring import local_dnds` without any `sys.path` hacking.

---

## Code Quality Requirements

- All major sections must be clearly marked with Markdown headers in the notebook
- Every function must have a docstring explaining its parameters and return values
- Use `tqdm` for progress bars during DB population and evaluation loops
- Wrap ChromaDB operations in try/except with informative error messages
- At the top of the notebook, include a single `CONFIG` dict for all hyperparameters:

```python
CONFIG = {
    "k_vote": 10,          # classification neighborhood size
    "K_density": 50,       # density estimation neighborhood size
    "alpha": 0.5,          # default image-text fusion weight
    "alpha_sweep": [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0],
    "imbalance_ratios": [10, 50, 100],
    "batch_size": 100,     # DB population batch size
    "epsilon": 1e-6,       # distance smoothing term
    "db_path": "./chroma_db",
    "figures_path": "./figures/phase2",
    "class_names": ["Black", "Blue", "Green", "TTR"],
    "minority_classes": ["TTR", "Green"],
    "majority_classes": ["Black", "Blue"],
}
```

---

## Final Results Summary

At the end of the notebook, print a clean summary table:

```
============================================================
PHASE 2 RAC EXPERIMENT — RESULTS SUMMARY
============================================================
Variant              | Accuracy | Macro F1 | TTR F1 | Latency
---------------------|----------|----------|--------|--------
Majority Vote        |   XX.X%  |  X.XXX   | X.XXX  | X.XX ms
IDW                  |   XX.X%  |  X.XXX   | X.XXX  | X.XX ms
Global DNDS          |   XX.X%  |  X.XXX   | X.XXX  | X.XX ms
Local DNDS (ours)    |   XX.X%  |  X.XXX   | X.XXX  | X.XX ms
Phase 1 Traditional  |   82.16% |  0.8177  | 0.8177 | 1.43 ms
============================================================
Best alpha (local DNDS): X.X
Best imbalance correction gain on TTR F1 at 100:1 ratio: +X.XX
```

---

## Project Structure

All Phase 2 code lives under the following structure. The agent must create files at exactly these paths.

```
trash-classification-project/
│
├── src/
│   └── phase2/
│       ├── __init__.py              # Exports key symbols from all modules
│       ├── db_client.py             # ChromaDB client init, collection setup, get_or_create logic
│       ├── embedders.py             # Image and text embedding wrappers (OpenCLIP + MiniLM)
│       ├── scoring.py               # All 5 scoring variants as standalone functions
│       ├── evaluation.py            # Metrics computation, classification report helpers
│       ├── imbalance.py             # Post-filtering logic for imbalance simulation
│       └── visualization.py         # All matplotlib/seaborn plotting functions
│
├── notebooks/
│   └── phase2/
│       ├── 01_db_population.ipynb       # Embed training set → populate both ChromaDB collections
│       ├── 02_evaluation.ipynb          # Implement all 5 scoring variants + full test set eval vs Phase 1
│       ├── 03_imbalance_experiment.ipynb # Imbalance simulation at 10:1 / 50:1 / 100:1
│       ├── 04_continual_learning.ipynb  # Progressive DB growth experiment
│       └── 05_results_summary.ipynb     # Aggregate all saved results, produce all final figures
│
├── results/
│   └── phase2/
│       ├── evaluation_results.json      # Saved outputs from 02_evaluation.ipynb
│       ├── imbalance_results.json       # Saved outputs from 03_imbalance_experiment.ipynb
│       └── continual_learning_results.json  # Saved outputs from 04_continual_learning.ipynb
│
├── figures/
│   └── phase2/
│       ├── scoring_comparison.png
│       ├── minority_f1_vs_imbalance.png
│       ├── alpha_sensitivity.png
│       ├── continual_learning_curve.png
│       ├── confusion_matrices_phase2.png
│       └── phase2_vs_phase1.png
│
└── chroma_db/                           # Persistent ChromaDB storage (gitignored)
```

### Module Responsibilities

**`db_client.py`**
- Initializes a `chromadb.PersistentClient` pointed at `CONFIG["db_path"]`
- Exposes `get_image_collection()` and `get_text_collection()` — both use `get_or_create_collection` so they are safe to call repeatedly
- Exposes `get_class_counts(collection)` — returns a `dict[str, int]` of label → count, used by `global_dnds`

**`embedders.py`**
- `ImageEmbedder` class: wraps OpenCLIP ViT-B/32, exposes `embed(image_path: str) -> np.ndarray` and `embed_batch(image_paths: list[str]) -> list[np.ndarray]`
- `TextEmbedder` class: wraps `all-MiniLM-L6-v2`, exposes `embed(text: str) -> np.ndarray` and `embed_batch(texts: list[str]) -> list[np.ndarray]`
- Both classes cache their model on first instantiation (singleton pattern)

**`scoring.py`**
- Each scoring variant is a standalone function with the same signature:
  ```python
  def score(
      query_image: np.ndarray,
      query_text: str,
      image_collection,
      text_collection,
      config: dict,
      alpha: float = 0.5
  ) -> str:   # returns predicted class label
  ```
- Functions: `majority_vote`, `idw`, `global_dnds`, `local_dnds`, `traditional`
- `traditional` additionally accepts `image_model`, `text_model`, `tokenizer`, `transform` as kwargs
- All variants use the same `CONFIG` dict for `k_vote`, `K_density`, `epsilon`

**`evaluation.py`**
- `evaluate_variant(score_fn, test_dataset, ...) -> dict` — runs inference over the full test set and returns the metrics dict
- `compute_metrics(y_true, y_pred, class_names) -> dict` — wraps sklearn metrics
- `save_results(results: dict, path: str)` and `load_results(path: str) -> dict` — JSON serialization for cross-notebook result sharing

**`imbalance.py`**
- `simulate_imbalance(results, majority_classes, minority_classes, ratio) -> filtered_results` — post-filters ChromaDB query results to enforce a given majority:minority ratio before scoring
- Takes raw ChromaDB result dicts as input so it slots cleanly between the `.query()` call and the scoring logic

**`visualization.py`**
- One function per figure, each accepting results dicts and saving to `figures/phase2/`
- All functions return the matplotlib `fig` object so the notebook can also call `plt.show()`

### Notebook Responsibilities

**`01_db_population.ipynb`**
- Imports `db_client`, `embedders`
- Loads train CSV and image paths
- Populates `trash_image_db` and `trash_text_db` in batches
- Prints final `collection.count()` for both collections to confirm success
- Idempotent: skips population if counts already match training set size

**`02_evaluation.ipynb`**
- Imports `scoring`, `evaluation`, `visualization`
- Runs all 5 variants on the full test set (no imbalance)
- Runs alpha sensitivity sweep for `local_dnds`
- Saves results to `results/phase2/evaluation_results.json`
- Generates: `scoring_comparison.png`, `alpha_sensitivity.png`, `confusion_matrices_phase2.png`, `phase2_vs_phase1.png`

**`03_imbalance_experiment.ipynb`**
- Imports `scoring`, `evaluation`, `imbalance`, `visualization`
- Runs all 5 variants at imbalance ratios `[10, 50, 100]`
- Saves results to `results/phase2/imbalance_results.json`
- Generates: `minority_f1_vs_imbalance.png`

**`04_continual_learning.ipynb`**
- Imports `db_client`, `scoring`, `evaluation`, `visualization`
- Progressively grows DB from 10% → 100% of training set in 10% increments
- At each step evaluates `local_dnds` on full test set
- Saves results to `results/phase2/continual_learning_results.json`
- Generates: `continual_learning_curve.png`

**`05_results_summary.ipynb`**
- Imports `evaluation`, `visualization`
- Loads all three JSON result files from `results/phase2/`
- Prints the final summary table (see "Final Results Summary" section)
- Re-generates all figures from saved data (no inference needed)
- Acts as the single source of truth for the paper's result section

### `.gitignore` additions

Add the following to the repo's `.gitignore`:
```
chroma_db/
results/phase2/
figures/phase2/
src/phase2/__pycache__/
```

---

## Notes

- Do NOT store or hardcode the dataset path — read it from a relative path consistent with the existing repo structure
- The dataset and model files are NOT in the repo (too large); the notebook should fail gracefully with a clear error if they are missing
- Maintain compatibility with the existing `uv` + `pyproject.toml` workflow
- The notebook should be runnable end-to-end with a single "Run All" without manual intervention (after DB is populated once)