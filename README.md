# <p align="center">ComicGTN infers rare cell states from single-cell multi-omics data using DNA sequence-augmented graph transformer networks.</p>
## Framework
`ComicGTN` is an innovative computational framework that integrates single-cell multi-omics data with DNA sequence information via an enhanced graph transformer networks to accurately identify rare target clusters.  
  
![Figure1-修](https://github.com/user-attachments/assets/670cb3b7-f7fd-4b1e-89c3-5ac95d3bc844)
## Requirements and Installation
### System requirements
Only a standard computer with sufficient RAM to support the in-memory operations is required to use the `ComicGTN` package. `ComicGTN` supports multiple operating systems: GNU/Linux, Windows, or Macs. If a GPU possessing enough VRAM is equipped, `ComicGTN` can also be deployed on it to accelerate computations.
### Dependencies of ComicGTN
[![numpy](https://img.shields.io/badge/numpy-V1.26.4-red?style=flat)](https://pypi.org/project/numpy/1.26.4/)
[![pandas](https://img.shields.io/badge/pandas-V2.0.3-orange?style=flat)](https://pypi.org/project/pandas/2.0.3/)
[![scipy](https://img.shields.io/badge/scipy-V1.11.4-yellow?style=flat)](https://pypi.org/project/scipy/1.11.4/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-V1.1.0-green?style=flat)](https://pypi.org/project/scikit-learn/1.1.0/)
[![torch](https://img.shields.io/badge/torch-V2.6.0-cyan?style=flat)](https://pypi.org/project/torch/2.6.0/)
[![torch-geometric](https://img.shields.io/badge/torch--geometric-V2.6.1-blue?style=flat)](https://pypi.org/project/torch-geometric/2.6.1/)
[![torch-scatter](https://img.shields.io/badge/torch--scatter-V2.1.2-purple?style=flat)](https://pypi.org/project/torch-scatter/2.1.2/)
[![torch-sparse](https://img.shields.io/badge/torch--sparse-V0.6.18-pink?style=flat)](https://pypi.org/project/torch-sparse/0.6.18/)
[![leidenalg](https://img.shields.io/badge/leidenalg-V0.10.2-silver?style=flat)](https://pypi.org/project/leidenalg/0.10.2/)
[![scanpy](https://img.shields.io/badge/scanpy-V1.10.4-gold?style=flat)](https://pypi.org/project/scanpy/1.10.4/)
[![anndata](https://img.shields.io/badge/anndata-V0.11.1-chocolate?style=flat)](https://pypi.org/project/anndata/0.11.1/)
[![matplotlib](https://img.shields.io/badge/matplotlib-V3.10.0-olive?style=flat)](https://pypi.org/project/matplotlib/3.10.0/)
[![seaborn](https://img.shields.io/badge/seaborn-V0.13.2-violet?style=flat)](https://pypi.org/project/seaborn/0.13.2/)
[![tqdm](https://img.shields.io/badge/tqdm-V4.67.1-lavender?style=flat)](https://pypi.org/project/tqdm/4.67.1/)
### Installation step
`ComicGTN` is developed in Python (version >= 3.10). We recommend creating a new environment and executing the following command to install `ComicGTN`:
```bash
# Create and activate a new Python environmnet
conda create -n comicgtn python=3.10
conda actiavte comicgtn
```
First clone the repository：
```bash
git clone https://github.com/Jinsl-lab/ComicGTN.git
cd ComicGTN
```
Then install the ComicGTN package by pip:
```bash
pip install -e .
```
## Download data
All datasets involved in the experiment can be obtained from the following source:
* [BMMC multi-omics dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE194122)
* [Mouse kidney multi-omics dataset](https://openproblems.bio/datasets/openproblems_v1_multimodal/scicar_mouse_kidney)
* [T-depleted BMMC multi-omics dataset](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE200046)
* [Mouse breast tumors](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE212707)
* [Frozen cortical tissues of MOGHE patients](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE284073)
* [Kidney organoid derived from ARPKD patients](https://www.ncbi.nlm.nih.gov/geo/query/acc.cgi?acc=GSE244608)
## Data preprocess
Prior to the analysis, first convert the TSV file containing peaks (open chromatin regions) into the .bed format, which stores genomic coordinates.
```python
import ComicGTN
from ComicGTN.utils import Peak_To_Bed
workdir = "./ComicGTN/data"
Peak_names = pd.read_csv(os.path.join(workdir, "Peak_names.tsv"), sep = "\t", header = None)
Peak_To_Bed(Peak_names, workdir)
```
Next, please install some essential dependencies to enable subsequent processing of DNA sequences (k-mers and TF motifs).
```python
os.chdir("./ComicGTN/data")
#! wget https://hgdownload.soe.ucsc.edu/goldenPath/hg19/bigZips/hg19.fa.gz
#! gunzip hg19.fa.gz
#! wget https://hgdownload.soe.ucsc.edu/goldenPath/mm10/bigZips/mm10.fa.gz
#! gunzip mm10.fa.gz
! wget https://raw.githubusercontent.com/pinellolab/simba/master/R_scripts/scan_for_kmers_motifs.R
```
```bash
conda install r-essentials r-optparse bioconductor-jaspar2020 bioconductor-biostrings bioconductor-tfbstools bioconductor-motifmatchr bioconductor-summarizedexperiment r-doparallel bioconductor-rhdf5 bioconductor-hdf5array
```
Finally, navigate to the /data directory and execute the following command to accomplish the scan of frequency matrices (k-mer-peak and motif-peak).
```bash
# For the human species
Rscript scan_for_kmers_motifs.R -i Peaks.bed -g hg19.fa -s "Homo sapiens"
# For the mouse species
Rscript scan_for_kmers_motifs.R -i Peaks.bed -g mm10.fa -s "Mus musculus"
```
## Tutorials
We provide a simple example demonstrating how to use ComicGTN to infer rare cell populations and visualize the results. Before starting this tutorial, please ensure you have completed all necessary steps outlined in README.md. If you encounter any issues during the process, feel free to contact us.  
[ComicGTN Tutorial on BMMC-bench-1](https://github.com/Jinsl-lab/ComicGTN/blob/main/tutorial/example.ipynb)
