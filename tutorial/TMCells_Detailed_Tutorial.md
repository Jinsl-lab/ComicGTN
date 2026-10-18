# TMCells: Detailed Tutorial

This tutorial describes how to install and apply **TMCells** to a new single-cell dataset. It covers **software installation**, followed by the five key workflow stages: **(1) preprocessing and input requirements, (2) running TMCells, (3) generating metacell assignments, (4) parameter settings, and (5) downstream analysis**.

This detailed tutorial includes executable Python code blocks and technical explanations. Users can copy the code into a local Python session or notebook and run it with their own data. 

For a shorter example, see [Quick Start](https://github.com/Jinsl-lab/TMCells/blob/main/tutorial/TMCells_QuickStart.ipynb).

## 1. Installation

Create a Python environment:

```bash
conda create -n tmcells python=3.11 -y
conda activate tmcells
```

Install dependencies (select a [PyTorch build](https://pytorch.org/get-started/locally/) compatible with your hardware if using a GPU):

```bash
python -m pip install torch numpy pandas scipy scikit-learn scanpy anndata leidenalg \
    matplotlib seaborn alive-progress psutil tensorboard jupyterlab
```

Clone the repository:

```bash
git clone https://github.com/Jinsl-lab/TMCells.git
cd TMCells
python -c "from TMCells import TMCells, UnsupervisedTrainer"
```

Run the following Python examples from the repository root. **The current repository has no Python packaging configuration**, so `pip install -e .` is not supported until a `pyproject.toml` or equivalent is added.

## 2. Input data and preprocessing

TMCells takes an `AnnData` object with a non-negative **cells × features** count matrix in `adata.X`. Reference cell-type labels are optional and are not used for model training. Provide a quality-controlled dataset appropriate for your assay. No extra normalization, log transformation, feature selection, or PCA is required before training.

```python
from pathlib import Path
import numpy as np
import pandas as pd
import anndata as ad
import scanpy as sc
from scipy import sparse

from TMCells import (
    TMCells, UnsupervisedTrainer, Metacell_Matrix,
    compute_celltype_purity, plot_metacell_sizes,
    plot_metacell_2D, plot_metacell_2D_bycelltype,
)

DATA_PATH = Path('path/to/your_data.h5ad')
adata = sc.read_h5ad(DATA_PATH)
CELLTYPE_KEY = 'cell_type' if 'cell_type' in adata.obs else None

# If counts are stored in a layer, use that layer as the model input.
# adata.X = adata.layers['counts'].copy()

# Required input checks only; no additional QC filters.
if adata.n_obs == 0 or adata.n_vars == 0:
    raise ValueError('The dataset is empty.')
X = adata.X
values = X.data if sparse.issparse(X) else np.asarray(X)
if not np.isfinite(values).all() or (values < 0).any():
    raise ValueError('Input counts must be finite and non-negative.')
if (np.asarray(X.sum(axis=1)).ravel() <= 0).any():
    raise ValueError('Remove or resolve zero-count cells before training.')
```

For a new dataset, change `DATA_PATH` and, if reference annotations exist, set `CELLTYPE_KEY` to the corresponding column name. Use `None` for an unlabeled dataset.

## 3. Run TMCells

The following code follows the [Quick Start](https://github.com/Jinsl-lab/TMCells/blob/main/tutorial/TMCells_QuickStart.ipynb).

```python
# Initialize the model.
model = TMCells(
    n_features=adata.n_vars,
    n_metacells=115,
    n_topics=15,
    seed=42,
)

# Train the model.
trainer = UnsupervisedTrainer(model, adata, batch_size=512, seed=42)
trainer.train(
    n_epochs=200,
    t_warmup=50,
    max_kl_weight=1e-4,
    save_model_ckpt=False,
)
```

`n_metacells=115` is an **example**, not a recommended value for every dataset. Training is unsupervised and does not use `CELLTYPE_KEY`.

## 4. Generate metacell assignments

```python
model.get_cell_embeddings_and_metacells(adata, batch_size=512)

# Save cell-to-metacell assignments and the annotated AnnData object.
out_dir = Path('tmcells_results')
out_dir.mkdir(exist_ok=True)
pd.DataFrame({
    'cell_id': adata.obs_names,
    'metacell_id': adata.obs['TMCells'].to_numpy(),
}).to_csv(out_dir / 'cell_to_metacell.csv', index=False)
adata.write_h5ad(out_dir / 'cells_with_tmcells.h5ad')
```

The results are stored in `adata.obs['TMCells']` (hard assignments), `adata.obsm['theta']` (topic proportions), and `adata.obsm['delta']` (latent representations). The assignment IDs **are not cell-type labels**.

## 5. Parameter settings

| Parameter | Example | Description |
|---|---:|---|
| `n_metacells` | 115 | Number of Gaussian mixture components; some may be empty |
| `n_topics` | 15 | Number of latent topics (model default: 15) |
| `batch_size` | 512 | Training or inference batch size |
| `n_epochs` | 200 | Maximum number of training epochs in this example |
| `max_kl_weight` | `1e-4` | Target KL-divergence weight |
| `t_warmup` | 50 | Number of epochs for linear KL-weight warm-up |
| `seed` | 42 | Random seed |

Choose `n_metacells` according to dataset size and desired resolution. Adjust `batch_size` for available memory. The library defaults may differ from the example values. Record the selected settings when reporting results.

## 6. Downstream analysis

These analyses are optional and do not change the metacell assignments.

### 6.1 Metacell size and visualization

```python
plot_metacell_sizes(adata, metacell_index='TMCells')

# Visualize metacells in the single-cell topic embedding space.
sc.pp.neighbors(adata, n_neighbors=15, use_rep='theta')
sc.tl.umap(adata)

if CELLTYPE_KEY is not None:
    plot_metacell_2D_bycelltype(
        adata, metacell_index='TMCells', celltype_key=CELLTYPE_KEY,
        key='X_umap', colour_metacells=True, title='TMCells',
        show=True, figsize=(5, 5), metacell_size=10, cell_size=5,
    )
else:
    plot_metacell_2D(adata, metacell_index='TMCells', key='X_umap', show=True)
```

Metacell positions are the mean UMAP coordinates of constituent cells, **not** a separate metacell-expression UMAP.

### 6.2 Construct the metacell expression matrix

Use the built-in function when reference annotations are available:

```python
if CELLTYPE_KEY is not None:
    mc_adata = Metacell_Matrix(
        adata, cell_type_label=CELLTYPE_KEY,
        metacell_label='TMCells', mc_mode='average',
    )
```

The current `Metacell_Matrix` function requires an annotation column because it also calculates post hoc majority labels and purity. For unlabeled data, average the same assignments directly:

```python
if CELLTYPE_KEY is None:
    codes, mc_ids = pd.factorize(adata.obs['TMCells'], sort=True)
    membership = sparse.csr_matrix(
        (np.ones(adata.n_obs), (codes, np.arange(adata.n_obs))),
        shape=(len(mc_ids), adata.n_obs),
    )
    sizes = np.bincount(codes)
    X_mc = sparse.diags(1 / sizes) @ (membership @ adata.X)
    mc_adata = ad.AnnData(
        X=X_mc,
        obs=pd.DataFrame(index=[str(i) for i in mc_ids]),
        var=adata.var.copy(),
    )

mc_adata.write_h5ad(out_dir / 'aggregated_metacells.h5ad')
```

Here `mc_adata.X` contains **average** feature values over cells in each metacell.

### 6.3 Purity assessment (labeled data only)

```python
if CELLTYPE_KEY is not None:
    purity = compute_celltype_purity(
        adata, celltype_label=CELLTYPE_KEY, metacell_label='TMCells',
    )
    purity.to_csv(out_dir / 'metacell_purity.csv')
```

Purity measures agreement with **existing annotations** and is not de novo cell-type identification.

### 6.4 Leiden clustering (resolution = 1)

```python
# Normalization is applied only to the aggregated metacell copy.
mc_down = mc_adata.copy()
sc.pp.normalize_total(mc_down, target_sum=1e4)
sc.pp.log1p(mc_down)

n_pcs = min(30, mc_down.n_obs - 1, mc_down.n_vars - 1)
if n_pcs < 2:
    raise ValueError('Too few metacells or features for PCA.')
sc.tl.pca(mc_down, n_comps=n_pcs)
sc.pp.neighbors(mc_down, n_neighbors=min(15, mc_down.n_obs - 1), use_rep='X_pca')
sc.tl.leiden(mc_down, resolution=1.0, key_added='leiden_r1')

# Propagate unsupervised metacell clusters to constituent cells.
cluster_map = mc_down.obs['leiden_r1'].astype(str)
adata.obs['leiden_r1_metacell'] = (
    adata.obs['TMCells'].astype(str).map(cluster_map)
)
```

With reference annotations, clustering agreement can be evaluated using AMI and ARI:

```python
if CELLTYPE_KEY is not None:
    from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score
    truth = adata.obs[CELLTYPE_KEY].astype(str)
    clusters = adata.obs['leiden_r1_metacell'].astype(str)
    ami = adjusted_mutual_info_score(truth, clusters)
    ari = adjusted_rand_score(truth, clusters)
```

### 6.5 Marker-gene expression (scRNA-seq example)

```python
if CELLTYPE_KEY is not None and 'cell_type' in mc_down.obs:
    genes = [g for g in ['INS', 'GCG', 'SST', 'PPY'] if g in mc_down.var_names]
    if genes:
        sc.pl.dotplot(mc_down, var_names=genes, groupby='cell_type')
```

Replace these example genes with markers for your dataset. For scATAC-seq, use an appropriate gene-activity representation rather than interpreting peak counts as gene expression.

## Notes

- The notebook and Markdown documentation do **not** need embedded training logs or saved figures. Code produces results when executed.
- Training uses the untransformed input matrix. Normalization in Section 6.4 applies **only to downstream metacell analysis**.
- For reproducibility, report the dataset, code version, random seed, and parameter settings.

Source code: [TMCells](https://github.com/Jinsl-lab/TMCells) · [metacell evaluation](https://github.com/Jinsl-lab/TMCells/blob/main/TMCells/metacell_eval.py) · [visualization](https://github.com/Jinsl-lab/TMCells/blob/main/TMCells/metacell_plot.py).
