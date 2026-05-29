***
<!-- <table>
  <tr>
    <td width="150">
      <img src="figures/assets/astra_logo_black.png" width="150" height="120" alt="ASTRA Logo">
    </td>
    <td>
      <h1>ASTRA:<sup><sub><b>A</b>ttention-based <b>S</b>elf-supervised <b>T</b>ime-series <b>R</b>epresentation <b>A</b>rchitecture</sub></sup></h1>
    </td>
  </tr>
</table> -->

<img src="figures/assets/astra_logo_black.png" width="150" height="140" align="left" alt="ASTRA Logo">

### ASTRA: <sup><sub><b>A</b><sub>ttention-based</sub> <b>S</b><sub>elf-supervised</sub> <b>T</b><sub>ime-series</sub> <b>R</b><sub>epresentation</sub> <b>A</b><sub>rchitecture</sub></sub></sup>

<br clear="left"/>

ASTRA is a self-supervised learning (SSL) framework designed for astronomy time-series data, specifically optimized for light curve analysis. By leveraging attention-based representation learning, ASTRA enables pre-training on unlabelled astronomical datasets to extract meaningful physical representations, which can then be fine-tuned for downstream classification and analysis tasks.

The framework currently supports **Astra-CLR** (our contrastive learning implementation) and is designed to support knowledge distillation models in future releases.

---

## 🌐 The ASTRA Ecosystem
To facilitate development, deployment, and research reproducibility, the ASTRA project is split across four distinct repositories:

| Repository | Description | Link |
| :--- | :--- | :--- |
| **Development Code** | Core framework, data pipeline, and model training (This Repo). | *[Current Repository]* |
| **Inference Code** | Lightweight, production-ready inference scripts and utilities. | [GitHub: astra-inference](https://github.com/your-username/astra-inference) *(Replace Link)* |
| **Astronomy Dataset** | Light curve datasets formatted for ASTRA, introduced here for the first time. | [Hugging Face: Dataset](https://huggingface.co/datasets/your-username/astra-dataset) *(Replace Link)* |
| **Model Weights & ONNX** | Pre-trained & finetuned model checkpoints along with ONNX exports. | [Hugging Face: Models](https://huggingface.co/models/your-username/astra-models) *(Replace Link)* |

---

## 🛠️ Installation & Environment Setup

Clone the repository and navigate to the root directory:
```bash
git clone https://github.com/TorshaMajumder/astra.git
cd astra
```

### MLflow Logging Setup *(Optional)*
If you require experiment tracking with MLflow, install these additional dependencies:
```bash
pip install mlflow
pip install pyarrow==22.0.0
```

---

## 1. Catalog Preparation (HATS Notebooks)

This phase prepares your astronomical catalogs by cross-matching the Gaia DR3 variability catalogs, PS1 DR2, and Zubercal DR16, and storing them in [HATS](https://hats.readthedocs.io) format. These HATS-formatted directories will subsequently be processed into Tensor Records for model training.

### Workflow
Please run the notebooks in the following order:

1. **`import-vizier.ipynb`** — Downloads and converts the source catalogs from VizieR into HATS format.
2. **`zubercal-x-gaia-vars.ipynb`** — Cross-matches the Zubercal DR16 catalog with the Gaia variable star catalog.

### Loading Examples
Notebooks prefixed with `_` demonstrate how to load the resulting HATS catalogs for validation or manual inspection:

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
The `astra-data` tool processes your prepared HATS catalogs into TFRecord chunks. Use the arguments below depending on your dataset structure.

* **Case A: Dataset contains a "Class" column for labels**
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
ASTRA-NET utilizes distributed training. This phase requires a separate virtual environment to manage dependencies like TensorFlow and Keras.

### Set up the Training Environment
```bash
python3.11 -m venv venv_train
source venv_train/bin/activate

# Optional: Upgrade package managers
pip install --upgrade pip setuptools wheel pip-tools
pip install -e .  
pip install -r train-requirements.txt 
```

### Run Model Training (`astra-net`)
You can train models using contrastive representation learning (Astra-CLR) or prepare for knowledge distillation workflows.

```bash
# Pre-training with Contrastive Loss (Astra-CLR)
astra-net --loss contrastive --config ../config/contrastive-loss_triplet.yaml --num_gpus 0 --epoch 100 --batch_size 300

# Pre-training with Knowledge Distillation (Future Framework Support)
astra-net --loss k_distil --config ../config/k_distil.yaml --num_gpus 0 --epoch 100 --batch_size 300
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

# For Knowledge Distillation framework (Run on 'main' branch once released)
astra-embeddings --loss k_distil --config ../config/k_distil-loss_embeddings.yaml --batch_size 500 
```

### Perform Downstream Tasks (`astra-downstream`)
Assess the quality of the learned representations on downstream astronomical classification or regression tasks:
```bash
astra-downstream --config ../config/downstream_task.yaml 
```

### Perform Supervised Finetuning (`astra-finetuning`)
Fine-tune the pre-trained weights end-to-end using labeled light curve data:
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

