# Enviroments and Configurations

## Requirements

```bash
conda create -n molca python=3.8
conda activate molca
conda install pytorch==2.0.1 torchvision==0.15.2 torchaudio==2.0.2 pytorch-cuda=11.7 -c pytorch -c nvidia
conda install pyg -c pyg
pip install rouge_score nltk ogb peft rdkit salesforce-lavis
pip install -U transformers pytorch-lightning
pip install deepspeed
```

Download nltk corpus:
```python
import nltk
nltk.download('wordnet')
```

## Dataset

Unzip `qm9.zip` under the `./data/` directory.

## Train

### Fine-tune Stage
Run the following script for fine-tuning on the PubChem324k dataset:

```bash
python main_graph_token_regression.py
```

> **Note**: Please download the checkpoints of Vicuna 1.5 from [this link](https://huggingface.co/lmsys/vicuna-7b-v1.5) and put them under the `./LLM` directory.

# Methods in LLM4graph
| Ablation Models       | default | Build | Algo | NodeSeq | CoT | Prefix | LoRA | Alignment | Position | Multi-level | Context | Category |
|-----------------------|---------|-------|------|---------|-----|--------|------|-----------|----------|-------------|---------|----------|
| Prompts Only          | ✓       |       |      |         |     |        |      |           |          |             |         | G2to     |
| GNN Only              |         |       |      |         |     |        |      | ✓         |          |             |         |          |
| **Standard Model**    | ✓       |       |      |         |     |        |      | ✓         |          |             |         | G2to     |
| Different Prompts 1   |         | ✓     |      |         |     |        |      | ✓         |          |             |         | G2te     |
| Different Prompts 2   |         |       | ✓    |         |     |        |      | ✓         |          |             |         | G2to     |
| Different Prompts 3   |         |       |      | ✓       |     |        |      | ✓         |          |             |         | G2to     |
| Different Prompts 4   |         |       |      |         | ✓   |        |      | ✓         |          |             |         | G2te     |
| Prompt+Prefix         | ✓       |       |      |         |     | ✓      |      | ✓         |          |             |         | G2to     |
| Prompt+LoRA           | ✓       |       |      |         |     |        | ✓    | ✓         |          |             |         | G2te     |
| Prompt+Position       | ✓       |       |      |         |     |        |      | ✓         | ✓        |             |         | G2te     |
| Prompt+Multi-level    | ✓       |       |      |         |     |        |      | ✓         |          | ✓           |         | G2te     |
| Prompt+Context        | ✓       |       |      |         |     |        |      | ✓         |          |             | ✓       | G2to     |
| **Full Model**        | ✓       |       |      |         |     | ✓      | ✓    | ✓         | ✓        | ✓           | ✓       |          |


Default Prompts

Build-a-Graph Prompting
