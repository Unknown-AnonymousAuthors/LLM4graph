# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.
import random
import torch
from pytorch_lightning import LightningDataModule
import torch_geometric
# from torch_geometric.loader import DataLoader
from torch_geometric.data import Data
from torch.utils.data import DataLoader, Dataset
from torch_geometric.loader.dataloader import Collater
from torch_geometric.transforms import AddLaplacianEigenvectorPE
import re
from rdkit import RDLogger, Chem
from data_process.smiles2graph_regression import smiles2graph
from data_process.text_attr import node_to_text, bond_to_text, get_chemberta_embeddings

RDLogger.DisableLog('rdApp.*')

class TrainCollater:
    def __init__(self, tokenizer, text_max_len):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])

    def __call__(self, batch):
        graph, instruction, text, text_values = zip(*batch)
        graphs = self.collater(graph)

        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(text=instruction,
                                            truncation=True,
                                            max_length=self.text_max_len,
                                            padding='longest',
                                            add_special_tokens=True,
                                            return_tensors='pt',
                                            return_attention_mask=True)

        text_values = torch.tensor(text_values).to(torch.float32)

        self.tokenizer.padding_side = 'right'
        text_tokens = self.tokenizer(text=text,
                                     truncation=True,
                                     padding='longest',
                                     add_special_tokens=True,
                                     max_length=self.text_max_len,
                                     return_tensors='pt',
                                     return_attention_mask=True)
        return graphs, instruction_tokens, text_tokens, text_values


class InferenceCollater:
    def __init__(self, tokenizer, text_max_len):
        self.text_max_len = text_max_len
        self.tokenizer = tokenizer
        self.collater = Collater([], [])

    def __call__(self, batch):
        graph, instruction, text, text_values = zip(*batch)
        graphs = self.collater(graph)
        # deal with prompt
        self.tokenizer.padding_side = 'right'
        self.tokenizer.truncation_side = 'left'
        instruction_tokens = self.tokenizer(instruction,
                                            return_tensors='pt',
                                            max_length=self.text_max_len,
                                            padding='longest',
                                            truncation=True,
                                            return_attention_mask=True)

        text_values = torch.tensor(text_values).to(torch.float32)
        return graphs, instruction_tokens, text, text_values    

def smiles2data(smiles, node_embedding, bond_embedding):
    graph = smiles2graph(smiles)
    x = torch.from_numpy(graph['node_feat'])
    edge_index = torch.from_numpy(graph['edge_index'], )
    edge_attr = torch.from_numpy(graph['edge_feat'])
    
    ''' !!! textual attributes '''
    if node_embedding is not None and bond_embedding is not None:
        node_embedding = torch.Tensor(node_embedding)
        bond_embedding = torch.Tensor(bond_embedding)
        bond_embedding = bond_embedding.repeat(1, 2).reshape(-1, bond_embedding.shape[1])
        
        node_embedding = torch.cat([x, node_embedding], dim=-1)
        bond_embedding = torch.cat([edge_attr, bond_embedding], dim=-1)
    
    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
    
    if False: # !!! graph PE
        transform = AddLaplacianEigenvectorPE(k=3, is_undirected=True)
        data = transform(data)
    
    return data


class CheBIDataset(Dataset):
    def __init__(self, path, text_max_len, prompt=None):
        self.path = path
        self.text_max_len = text_max_len
        self.prompt = prompt

        with open(self.path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            lines = [line.strip() for line in lines][1:]
        
        if False: # !!! textual embedding
            node_embeddings_list, bond_embeddings_list = torch.load(path[:-4]+"_textual.pt")
        else:
            node_embeddings_list = [None for _ in range(len(lines))]
            bond_embeddings_list = [None for _ in range(len(lines))]

        assert len(lines) == len(node_embeddings_list) and len(lines) == len(bond_embeddings_list)
        
        tuple_list = [(line, node_embedding, bond_embedding) for line, node_embedding, bond_embedding in zip(lines, node_embeddings_list, bond_embeddings_list)]
        
        # !!! sample
        random.seed(666)
        list_tuple = random.sample(tuple_list, int(len(lines) * 0.01))

        self.smiles_list = []
        self.instruction_list = []
        self.text_list = []
        self.node_embedding_list = []
        self.bond_embedding_list = []
        for line, node_embedding, bond_embedding in list_tuple:
            instruction, smiles, text = line.split('\t')  # qm9
            self.smiles_list.append(smiles)
            if True: # !!! Prompts
                instruction_a = f'''
                    Given the SMILES string "{smiles}".
                    Let’s construct a molecular graph with atoms as nodes and chemical bonds as edges.
                    Then, predict the HOMO energy value of this molecule.
                ''' # Build-a-Graph Prompting
                instruction_b = f'''
                    Given the SMILES string "{smiles}".
                    The idea is to start at one atom and use a graph traversal method (like message passing) to aggregate information from its bonded atoms.
                    At each atom, collect features like atom type and bond type.
                    After aggregating features across the molecule, predict the HOMO energy value.
                ''' # Algorithmic Prompting
                instruction_c = f'''
                    A chat between a curious user and an artificial intelligence assistant.
                    The assistant gives helpful, detailed, and polite answers to the user’s questions.
                    USER: Given the molecule described by the SMILES node sequence "{smiles}".
                    Please tell me the HOMO energy value of this molecule.
                ''' # Node Sequence Prompting (LLaGA)
                instruction_d = f'''
                    Given the SMILES string "{smiles}".
                    Please predict the HOMO energy value of this molecule in a step-by-step manner:
                    1. Parse the SMILES to identify atoms and bonds.
                    2. Analyze the molecular structure (e.g., functional groups, rings).
                    3. Estimate the HOMO energy of this graph based on structural features.
                ''' # Chain-of-Thought (CoT) Prompting
                instruction = instruction_a
            else:
                instruction = instruction # Graph2Token Prompting: The HOMO (Highest Occupied Molecular Orbital) represents the highest energy level in a molecule that contains electrons. Could you give me the HOMO energy value of this molecule?
            self.instruction_list.append(instruction)
            self.text_list.append(text)
            self.node_embedding_list.append(node_embedding)
            self.bond_embedding_list.append(bond_embedding)

    def __len__(self):
        return len(self.smiles_list)

    def __getitem__(self, index):
        smiles = self.smiles_list[index]
        instruction = self.instruction_list[index]
        text_values = float(self.text_list[index])
        text = self.text_list[index]
        node_embedding = self.node_embedding_list[index]
        bond_embedding = self.bond_embedding_list[index]
        
        graph = smiles2data(smiles, node_embedding, bond_embedding)
        return graph, instruction, text, text_values


class ProcessCheBIDM(LightningDataModule):
    def __init__(
            self,
            mode: str = 'pretrain',
            num_workers: int = 0,
            batch_size: int = 256,
            root: str = 'data/',
            text_max_len: int = 128,
            tokenizer=None,
            args=None,
    ):
        super().__init__()
        self.args = args
        self.mode = mode
        self.batch_size = batch_size
        self.inference_batch_size = args.inference_batch_size
        self.num_workers = num_workers
        self.text_max_len = text_max_len
        self.prompt = args.prompt
        if self.mode == 'pretrain':
            self.train_dataset = CheBIDataset(root + f'/train_sample_3600K.txt', text_max_len, self.prompt)
            self.val_dataset = CheBIDataset(root + '/valid.txt', text_max_len, self.prompt)
            self.test_dataset = CheBIDataset(root + '/test.txt', text_max_len, self.prompt)
        else:
            self.train_dataset = CheBIDataset(root + f'/train.txt', text_max_len, self.prompt)
            self.val_dataset = CheBIDataset(root + '/valid.txt', text_max_len, self.prompt)
            self.test_dataset = CheBIDataset(root + '/test.txt', text_max_len, self.prompt)
        self.init_tokenizer(tokenizer)

    def init_tokenizer(self, tokenizer):
        self.tokenizer = tokenizer
        self.train_dataset.tokenizer = tokenizer
        self.val_dataset.tokenizer = tokenizer
        self.test_dataset.tokenizer = tokenizer

    def train_dataloader(self):
        # assert self.mode == 'pretrain'
        # assert self.mode == 'ft'
        loader = DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=True,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
        )
        return loader

    def val_dataloader(self):
        val_loader = DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=TrainCollater(self.tokenizer, self.text_max_len),
        )

        return val_loader

    def test_dataloader(self):
        loader = DataLoader(
            self.test_dataset,
            batch_size=self.inference_batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=False,
            drop_last=False,
            persistent_workers=True,
            collate_fn=InferenceCollater(self.tokenizer, self.text_max_len),
        )
        return loader

    def add_model_specific_args(parent_parser):
        # !!! batch
        parser = parent_parser.add_argument_group("Data module")
        parser.add_argument('--num_workers', type=int, default=6) # 6
        parser.add_argument('--batch_size', type=int, default=1) # 16
        parser.add_argument('--inference_batch_size', type=int, default=1) # 16
        parser.add_argument('--use_smiles', action='store_true', default=False)
        parser.add_argument('--root', type=str, default='data/finetune/qm9/homo')
        # parser.add_argument('--root', type=str, default='data')
        parser.add_argument('--text_max_len', type=int, default=128) # 128
        parser.add_argument('--filtered_cid_path', type=str, default=None)
        parser.add_argument('--graph_only', action='store_true', default=False)
        return parent_parser


