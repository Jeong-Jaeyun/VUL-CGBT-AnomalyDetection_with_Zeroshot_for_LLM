from pathlib import Path

import torch
import torch.nn.functional as F
from gensim.models import KeyedVectors
from torch_geometric.data import Batch, Data, DataLoader
from torch_geometric.nn import GCNConv, global_add_pool, global_max_pool, global_mean_pool


_WORD2VEC_MODEL = None
_MODEL_PATH = Path(__file__).resolve().parents[1] / "data" / "GoogleNews-vectors-negative300.bin.gz"
DEFAULT_WORD2VEC_DIM = 300


def get_word2vec_model():
    global _WORD2VEC_MODEL
    if _WORD2VEC_MODEL is None:
        if not _MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Missing Word2Vec model file: {_MODEL_PATH}"
            )
        _WORD2VEC_MODEL = KeyedVectors.load_word2vec_format(str(_MODEL_PATH), binary=True)
    return _WORD2VEC_MODEL


def text_to_embedding(text, model=None):
    model = model or get_word2vec_model()
    if text is None:
        return torch.zeros(model.vector_size, dtype=torch.float32)
    words = text.split()    
    embeddings = [model[word] for word in words if word in model]    
    if embeddings:    
        return torch.tensor(sum(embeddings) / len(embeddings), dtype=torch.float32)
    else:    
        return torch.zeros(model.vector_size, dtype=torch.float32)
  
# Define GCN model    
class GCN(torch.nn.Module):    
    def __init__(self, in_channels, hidden_channels, out_channels, pooling_type="mean"):    
        super(GCN, self).__init__()  
        self.conv1 = GCNConv(in_channels, hidden_channels)    
        self.conv2 = GCNConv(hidden_channels, out_channels)    
        self.pooling_type = pooling_type
        
    def forward(self, x, edge_index, batch): 
        x = x.float()  
        edge_index = edge_index.long()    
        x = self.conv1(x, edge_index)    
        x = F.relu(x)    
        x = self.conv2(x, edge_index)   
        if self.pooling_type == "mean":
            x = global_mean_pool(x, batch)  # [num_graphs, out_channels]  
        elif self.pooling_type == "max":
            x = global_max_pool(x, batch)  # [num_graphs, out_channels] 
        elif self.pooling_type == "joint":
            x = global_add_pool(x, batch) * global_max_pool(x, batch)
        else:
            x = global_mean_pool(x, batch)
        #x = x.long()
        return x 
  
# Function to build a list of Data objects for CFG graphs  
def build_cfg_data_list(cfg_nodes_list, cfg_edges_list):    
    data_list = []    
    for nodes_text, edge_index in zip(cfg_nodes_list, cfg_edges_list):    
        # Ensure node features are float tensors  
        if nodes_text.ndim == 1:  
            nodes_text = nodes_text.unsqueeze(0)  
        data = Data(x=nodes_text, edge_index=edge_index)  
        data_list.append(data)    
  
    batch = Batch.from_data_list(data_list)    
    return batch  
  
# Function to build a list of Data objects for DFG graphs  
def build_dfg_data_list(dfg_nodes_list, dfg_edges_list):    
    data_list = []    
    for nodes_text, edge_index in zip(dfg_nodes_list, dfg_edges_list):    
        # Ensure node features are float tensors  
        if nodes_text.ndim == 1:  
            nodes_text = nodes_text.unsqueeze(0)  
        data = Data(x=nodes_text, edge_index=edge_index)  
        data_list.append(data)    
  
    batch = Batch.from_data_list(data_list)    
    return batch  
