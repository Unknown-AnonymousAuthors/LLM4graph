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


# Core Codes
#### Default Prompts
```python
#  (Line 157 in data_process/process_dm_regression.py, actually encoded in dataset files data/finetune/qm9/homo.zip)
instruction = '''
    The HOMO (Highest Occupied Molecular Orbital) represents the highest energy level in a molecule that contains electrons. Could you give me the HOMO energy value of this molecule? "{smiles}"
'''
```

#### Build-a-Graph Prompting
```python
#  (Line 131 in data_process/process_dm_regression.py)
instruction_a = f'''
    Given the SMILES string "{smiles}".
    Let’s construct a molecular graph with atoms as nodes and chemical bonds as edges.
    Then, predict the HOMO energy value of this molecule.
'''
```

#### Algorithmic Prompting
```python
#  (Line 137 in data_process/process_dm_regression.py)
instruction_b = f'''
    Given the SMILES string "{smiles}".
    The idea is to start at one atom and use a graph traversal method (like message passing) to aggregate information from its bonded atoms.
    At each atom, collect features like atom type and bond type.
    After aggregating features across the molecule, predict the HOMO energy value.
'''
```

#### Node Sequence Prompting
```python
#  (Line 142 in data_process/process_dm_regression.py)
instruction_c = f'''
    A chat between a curious user and an artificial intelligence assistant.
    The assistant gives helpful, detailed, and polite answers to the user’s questions.
    USER: Given the molecule described by the SMILES node sequence "{smiles}".
    Please tell me the HOMO energy value of this molecule.
'''
```

#### Chain-of-Thought Prompting
```python
#  (Line 149 in data_process/process_dm_regression.py)
instruction_d = f'''
    Given the SMILES string "{smiles}".
    Please predict the HOMO energy value of this molecule in a step-by-step manner:
    1. Parse the SMILES to identify atoms and bonds.
    2. Analyze the molecular structure (e.g., functional groups, rings).
    3. Estimate the HOMO energy of this graph based on structural features.
'''
```

#### Prefix fine-tuning
```python
#  (Line 144 in data_process/process_dm_regression.py)
prefix_embeds = self.prefix_embedding.expand(batch_size, -1, -1)
instruction_embeds = torch.cat([prefix_embeds, instruction_embeds], dim=1)
prefix_mask = torch.ones(batch_size, self.prefix_length, dtype=torch.long).to(device)
```

#### LoRA
```python
#  (Line 99 in model/QA_llama.py)
if self.lora_tuning:
    self.llm_model = PeftModel.from_pretrained(self.llm_model, peft_dir, is_trainable=True)
else:
    ...
```

#### The Alignment Problem
```python
#  (Line 138 in model/QA_llama.py)
instruction_embeds = self.llm_model.get_input_embeddings()(instruction_tokens.input_ids)

...

graph_inputs_llm = h_graph
graph_inputs_llm = graph_inputs_llm.unsqueeze(1)

llama_word_embeddings = self.word_embeddings.unsqueeze(0)
llama_word_embeddings = llama_word_embeddings.repeat(
    graph_inputs_llm.size(0), 1, 1).to(device)

cat_embedding = torch.cat([graph_inputs_llm, llama_word_embeddings], dim=1)
cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()
cat_embedding = self.mapping(cat_embedding)
cat_embedding = cat_embedding.permute(0, 2, 1).contiguous()

inputs_embeds = torch.cat([instruction_embeds, cat_embedding], dim=1)
```

#### The Position Problem
```python
# (Line 391 in model/gin_model.py)
pos = torch.arange(x.size(0), device=x.device)
pe = self.pe(pos)
x = x + pe

# (Line 92 in data_process/process_dm_regression.py)
transform = AddLaplacianEigenvectorPE(k=3, is_undirected=True)
data = transform(data)
```

#### The Multi-level Semantics Problem
```python
#  (Line366  in model/gin_model.py)
elif gnn_type == "gin_mixhop":
    self.gnns.append(GIN_MixHop(emb_dim))
```

#### The Context Problem
```python
#  (Line 81 in data_process/process_dm_regression.py)
if node_embedding is not None and bond_embedding is not None:
    node_embedding = torch.Tensor(node_embedding)
    bond_embedding = torch.Tensor(bond_embedding)
    bond_embedding = bond_embedding.repeat(1, 2).reshape(-1, bond_embedding.shape[1])
    
    x = torch.cat([x, node_embedding], dim=-1)
    edge_attr = torch.cat([edge_attr, bond_embedding], dim=-1)
```
