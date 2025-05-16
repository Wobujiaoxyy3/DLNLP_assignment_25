## Project Overview
This project implements an ontology-guided named entity recognition (NER) system for scientific text in the materials science domain. It leverages the MaterioMiner dataset, which provides both coarse-grained and fine-grained NER annotations based on a materials mechanics ontology.

We evaluate two transformer-based language models:

- **BERT**: A general-purpose pretrained language model (bert-base-uncased)

- **MatSciBERT**: A domain-specific model trained on materials science literature (m3rg-iitd/matscibert)

Our goal is to compare how ontology-based annotation and domain-specific pretraining impact NER performance in scientific contexts.

## Project Structure
``` bash
my_repo/
├── A/
│   ├── config.py
│   ├── dataloader.py
│   ├── get_dataset.py
│   ├── model.py
│   ├── trainer.py
│   └── utils.py
├── Datasets/
│   ├── coarse_grained_ner/
│   ├── fine_grained_ner/
│   └── data/
│  
├── main.py
├── README.md
└── requirements.txt

```

## Environment Setup

To get started with this NLP project, we recommend using a virtual environment to manage dependencies.

### 1. Create and activate a Python 3.8 virtual environment

Using conda:
```bash
conda create -n nlp_env python=3.8
conda activate nlp_env
```

### 2.Install required dependencies
Make sure you are in the project root directory (where `requirements.txt` is located), then run:
``` bash
pip install -r requirements.txt
```

**And You Are All Set！**

## Run the Project

### 1. Download Datasets
If you want to download the required datasets yourself, please go to [Datasets repo](https://gitlab.cc-asp.fraunhofer.de/iwm-micro-mechanics-public/ontologies/materials-mechanics-ontology) and copy the two folders in "dataset" folder, plus the two folders in `"example/data"`. You also need to store the files in the `Datasets` folder following the same structure as this repository. There's no need to worry about the other files in the `Datasets` folder — they will be automatically generated when you run `main.py`.

### 2. Run training and evaluation
Run the main training pipeline with:
``` bash
python main.py
```

You can also modify parameters by passing arguments through the command line when running `main.py`. The available parameters and options can be found in the args definition inside `main.py`.
For example, to change the model to `BERT` and increase the number of training epochs to `6`, you can run:
``` bash
python main.py --model_name matscibert --num_epochs 6
```

This allows you to easily customize the training configuration without modifying the source code directly.

## 📚 Acknowledgements

This project draws on resources from the Fraunhofer Institute's open initiatives. Specifically, we reference and build upon:

- [Materials Mechanics Ontology](https://gitlab.cc-asp.fraunhofer.de/iwm-micro-mechanics-public/ontologies/materials-mechanics-ontology):  
  A domain ontology for materials fatigue and microstructure-property modeling, licensed under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/).

- [MaterioMiner Dataset](https://gitlab.cc-asp.fraunhofer.de/iwm-micro-mechanics-public/datasets/materio-miner):  
  An ontology-guided named-entity recognition (NER) dataset and utility package for materials science publications.

All reused or adapted content respects the terms of the [Creative Commons Attribution 4.0 International License (CC BY 4.0)](https://creativecommons.org/licenses/by/4.0/), including proper attribution.

