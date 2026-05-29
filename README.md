***
<table>
  <tr>
    <td width="150">
      <img src="figures/assets/astra_logo_black.png" width="150" height="140" alt="ASTRA Logo">
    </td>
    <td>
      <h1>ASTRA: Attention-based Self-supervised Time-series Representation Architecture</h1>
    </td>
  </tr>
</table>
<p align="left">
  <img src="https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/Framework-TensorFlow%202.15-FF6F00?logo=tensorflow&logoColor=white" alt="TensorFlow Version">
  <img src="https://img.shields.io/badge/Hardware-NVIDIA%20A100-76B900?logo=nvidia&logoColor=white" alt="Hardware Version">
</p>

**ASTRA** is a self-supervised learning (SSL) framework designed for photometric time-series analysis. At its core, the framework utilizes **AstraNet**--an encoder model based on the original Transformer architecture--to extract robust representation vectors from raw light curves.
 

Our pre-training pipeline utilizes multi-filter light curves from the Zwicky Transient Facility (ZTF; [Bellm et al., 2019](https://ui.adsabs.harvard.edu/abs/2019PASP..131a8002B/abstract)) [Zubercal DR16](http://atua.caltech.edu/ZTF/Zubercal.html) catalog, prepared through a systematic cross-matching workflow against the [Gaia DR3](https://www.cosmos.esa.int/web/gaia/dr3) variability catalog.

Currently featuring **Astra-CLR** (our contrastive learning implementation), the framework also evaluates representations using a label-efficient, partial top-layer fine-tuning powered by a **Multi-View Late Fusion** mechanism. **ASTRA** is architected to seamlessly support knowledge distillation models in upcoming releases. 🚀


---

## 🌐 The ASTRA Ecosystem
To facilitate development, deployment, and research reproducibility, the ASTRA project is split across four distinct repositories:

| Repository | Description | Link |
| :--- | :--- | :--- |
| **Development Code** | Core framework, data pipeline, and model training (This Repo). | *[Current Repository]* |
| **Inference Code** | Lightweight, production-ready inference scripts and utilities. | [GitHub: astra-inference](https://github.com/snad-space/astra-infer) |
| **Astronomy Dataset** | Light curve datasets formatted for ASTRA, introduced here for the first time. | [Hugging Face: Dataset](https://huggingface.co/datasets/snad-space/astra-zubercaldr16_gaiadr3vclassre) |
| **Model Weights & ONNX** | Pre-trained & finetuned model checkpoints along with ONNX exports. | [Hugging Face: Models](https://huggingface.co/ashrot/astra-clr-base) |

---

## 🛠️ Installation & Environment Setup

Clone the repository and navigate to the root directory:
```bash
git clone https://github.com/TorshaMajumder/astra.git
cd astra
```

#### MLflow Logging Setup *(Optional)*
If you require experiment tracking with MLflow, install these additional dependencies:
```bash
pip install mlflow
pip install pyarrow==22.0.0
```

---

## 1. Catalog Preparation (HATS Notebooks)

This phase prepares your astronomical catalogs by cross-matching the Gaia DR3 variability catalogs, PS1 DR2, and Zubercal DR16, and storing them in [HATS](https://hats.readthedocs.io) format. These HATS-formatted directories will subsequently be processed into Tensor Records for model training.

### Workflow
Please run the notebooks under the directory [notebooks/hats/](notebooks/hats/) in the following order:

1. **`import-vizier.ipynb`** — Downloads and converts the source catalogs from VizieR into HATS format.
2. **`zubercal-x-gaia-vars.ipynb`** — Cross-matches the Zubercal DR16 catalog with the Gaia variable star catalog.

### Loading Examples
Notebooks inside the directory [notebooks/hats/](notebooks/hats/), prefixed with `_` demonstrate how to load the resulting HATS catalogs for validation or manual inspection:

* `_load-with-lsdb.ipynb` — Loading using [LSDB](https://docs.lsdb.io)
* `_load-with-nested-pandas.ipynb` — Loading using [nested-pandas](https://nested-pandas.readthedocs.io)

---

## 2. Data Pipeline: Creating Tensor Records
Once the HATS catalogs are prepared, use the data pipeline to generate TensorFlow Records (TFRecords) for model consumption.

### Set up the Data Environment
```bash
python3.11 -m venv venv_data
source venv_data/bin/activate

# Optional: Upgrade package managers
pip install --upgrade pip setuptools wheel pip-tools
pip install -e .  
pip install -r data-requirements.txt  
```

### Run Data Generation (`astra-data`)
The `astra-data` tool processes your prepared HATS catalogs into TFRecord chunks, optimizing data throughput and enabling highly efficient data ingestion for the training framework. 

* **Case A: Dataset contains a "Class" column to extract object label**
  ```bash
  astra-data --dest ../dataset/lyrae/ --path_to_buff ../dataset/lyrae/hats/zubercal_vrrlyr --min_detec 200 --train_size 0.80 --max_lcs_per_chunk 200
  ```

* **Case B: Dataset does not have a "Class" column (or requires renaming)**
  ```bash
  astra-data --dest ../dataset/agn/ --path_to_buff ../dataset/agn/hats/zubercal_vagn --min_detec 200 --train_size 0.80 --max_lcs_per_chunk 200 --label "AGN"
  ```

* **Case C: Removing specific classes from the dataset**
  ```bash
  astra-data --dest ../dataset/cepheids/ --path_to_buff ../dataset/cepheids/hats/zubercal_vcep --min_detec 200 --train_size 0.80 --max_lcs_per_chunk 200 --del_label ACEP DCEP
  ```

* **Case D: Keeping only specific classes from the dataset**
  ```bash
  astra-data --dest ../dataset/cepheids/ --path_to_buff ../dataset/cepheids/hats/zubercal_vcep --min_detec 200 --train_size 0.80 --max_lcs_per_chunk 200 --keep_label ACEP DCEP T2CEP
  ```

---

## 3. Model Training & Pre-training
**Astra-CLR** utilizes distributed training. This phase requires a separate virtual environment to manage dependencies like TensorFlow and Keras.

### Set up the Training Environment
```bash
python3.11 -m venv venv_train
source venv_train/bin/activate

# Optional: Upgrade package managers
pip install --upgrade pip setuptools wheel pip-tools
pip install -e .  
pip install -r train-requirements.txt 
```

### Run Model Training
Train representation learning models using the **Astra-CLR** contrastive learning framework. Please note that knowledge distillation workflows are not supported in the current release and will be integrated in a future update.

```bash
# Pre-training with Contrastive Loss (Astra-CLR)
# Note: --batch_size is defined per-replica (per-GPU)
astra-net --loss contrastive --config ../config/contrastive-loss_triplet.yaml --num_gpus 8 --epoch 200 --batch_size 78
```

---

## 4. Post-Training & Evaluation Workflows

⚠️ **Branch Note:** The main development branch holds the latest updates, including the upcoming knowledge distillation architecture. Because architectural changes in the distillation model modify the core codebase, downstream evaluations and fine-tuning for the contrastive learning framework must be executed from the dedicated `astra-clr` branch. 

Before running downstream tasks or contrastive model fine-tuning, please switch to the correct branch:
```bash
git checkout astra-clr
```

Once on the appropriate branch, you can proceed with the workflows below:

### Generate ASTRA Embeddings (`astra-embeddings`)
Generate representation embeddings for further statistical analysis or visual plotting:
```bash
# For Contrastive Loss framework (Run on 'astra-clr' branch)
astra-embeddings --loss contrastive --config ../config/contrastive-loss_embeddings.yaml --batch_size 500 

```

### Perform Downstream Tasks (`astra-downstream`)
Evaluate the quality of the learned representation vectors on downstream astronomical tasks. The framework currently supports:

* **Classification:** Evaluated via linear probing and weighted k-NN classification.
* **Anomaly Detection:** Evaluated via outlier detection using Isolation Forests.

```bash
astra-downstream --config ../config/downstream_task.yaml 
```

### Perform Top-Layer Finetuning (`astra-finetuning`)
Evaluate model performance in low-data regimes by dynamically sampling just **2%** of the labeled dataset. This pipeline performs partial top-layer fine-tuning utilizing a **Multi-View Late Fusion mechanism**--introducing this multi-view fusion architecture to astronomical time-series analysis.

```bash
astra-finetuning --loss contrastive --config ../config/contrastive-loss_finetuning.yaml 
```
---

## 🔍 Command Line Reference & Help

For detailed usage information and parameter descriptions for any of the CLI tools, run the commands below with the `--help` flag:

```bash
astra-data --help
astra-net --help
astra-embeddings --help
astra-downstream --help
astra-finetuning --help
```

***

## 🤝 Acknowledgments

We gratefully acknowledge the support and resources provided by the following organizations:
* **NVIDIA** — For hardware support and GPU computing resources (NVIDIA A100).
* **[Funding Body / Grant Name]** — [Placeholder].
* **[Institution / Department]** — [Placeholder].
* **[Institution / Department]** — [Placeholder].
 
*** 

