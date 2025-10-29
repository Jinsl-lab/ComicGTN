import os
import re
import math
import torch
import numpy as np
import pandas as pd
import scanpy as sc
import anndata as ad
import scanpy.external as sce
import torch.nn.functional as F
from tqdm import tqdm
from collections import Counter
from torch_scatter import scatter_add
from sklearn.metrics import confusion_matrix, f1_score, matthews_corrcoef, cohen_kappa_score
from .settings import Segment_Function


def Remove_Scaffold(ATAC_count, peak_names_data):
    non_scaffold_ids = []
    
    for i in range(len(peak_names_data)):
        if peak_names_data.iloc[i, 0][:3] == "chr":
            non_scaffold_ids.append(i)
    ATAC_count_re = ATAC_count[non_scaffold_ids, :]
    peak_names_data_re = peak_names_data.iloc[non_scaffold_ids, 0]
    
    # Peak--Cell count matrix and peak name list with scaffold removed.
    return ATAC_count_re, peak_names_data_re


def Peak_To_Bed(peak_names_data, workdir, filename = None):
    print("Peaks are being written to the '.bed' file, which may take a little time.")
    
    peak_names_data = peak_names_data[0].tolist()
    peak_df = pd.DataFrame(columns = ["chr", "start", "end"])
    for i in range(len(peak_names_data)):
        peak_split_array = np.array([re.split(r"[:-]",peak_names_data[i])])
        peak_df.loc[i] = peak_split_array[0]
    if filename is None:
        filename = os.path.join(workdir, "Peaks.bed")
    peak_df.to_csv(filename, sep = "\t", header = False, index = False)
    fp, fn = os.path.split(filename)
    
    # A .bed file containing three columns: chromosome name, start position, and end position.
    print(f"'{fn}' has been written to '{fp}'.")


def Subgraph_Segmentation(graph, seed, neighbor_node_num, node_select_prob):
    
    # Retrieve the specific type of neighbor node ID for the current Cell/Peak node.
    initial_selected_node = {seed}
    neighbor_node = graph[list(initial_selected_node), :].nonzero()[1]
    
    if len(neighbor_node) == 0:
        return []
    else:
        
        # The prop value of the neighbor node.
        neighbor_node_prop = node_select_prob[list(neighbor_node)]
        
        # The number of neighbor nodes = min(the number of nodes connected to the current Cell/Peak node, 20)
        real_neighbor_node_num = min(neighbor_node_num, len(neighbor_node))
        
        # Perform the softmax to calculate the prob value for node selection.
        prob = np.exp(neighbor_node_prop) / np.exp(neighbor_node_prop).sum()
        selected_neighbor_node = np.random.choice(neighbor_node, size = real_neighbor_node_num, replace = False, p = prob).tolist()
        node_list = sorted(selected_neighbor_node)
    
    # List of selected specific type neighbor node IDs.
    return node_list
    
    
def Subgraph_Extraction(RNA_count, ATAC_count, Kmer_count, Motif_count, neighbor_node_num = [20,20,10,3], cell_node_num = 30):
    print("Subgraph extraction is underway, and the running time is shown in the progress bar.")
    cell_node_id = np.random.choice(RNA_count.shape[1], size = RNA_count.shape[1], replace = False).tolist()
    batch_num = math.ceil(len(cell_node_id) / cell_node_num)
    
    # Used for storing the complete subgraph structure.
    total_node_list = []
    dic_cell = {}
    dic_peak = {}
    cell_node_no_gene_list = []
    cell_node_no_peak_list = []
    peak_node_no_kmer_list = []
    peak_node_no_motif_list = []
    
    # Ensure that the original count matrix remains unmodified during execution.
    RNA_count_copy = RNA_count
    ATAC_count_copy = ATAC_count
    Kmer_count_copy = Kmer_count.copy()
    Motif_count_copy = Motif_count.copy()
    
    for i in tqdm(range(batch_num)):
        gene_node_list = []
        peak_node_list = []
        kmer_node_list = []
        motif_node_list = []
        peak_node_list_non_repeat = []
        
        for index_cell, cell_node in enumerate(cell_node_id[i * cell_node_num:(i + 1) * cell_node_num]):
            gene_exp = RNA_count_copy[:, cell_node].todense()
            peak_exp = ATAC_count_copy[:, cell_node].todense()
            gene_exp[gene_exp < 5] = 0
            gene_node_list_temp = Subgraph_Segmentation(RNA_count.transpose(), cell_node, neighbor_node_num[0], 
                                                                                            np.squeeze(np.array(np.log(gene_exp + 1))))      
            peak_node_list_temp = Subgraph_Segmentation(ATAC_count.transpose(), cell_node, neighbor_node_num[1], 
                                                                                            np.squeeze(np.array(np.log(peak_exp + 1)))) 
            if gene_node_list_temp == []:
                cell_node_no_gene_list.append(cell_node)
            if peak_node_list_temp == []:
                cell_node_no_peak_list.append(cell_node)
            dic_cell[cell_node] = {"g": gene_node_list_temp, "p": peak_node_list_temp}
            ids = len(peak_node_list_non_repeat)
            
            # Summarize the IDs of the neighboring Gene nodes of all Cell nodes in the current subgraph.
            gene_node_list = gene_node_list + gene_node_list_temp
            
            # Summarize the IDs of the neighboring Peak nodes of all Cell nodes in the current subgraph.
            peak_node_list = peak_node_list + peak_node_list_temp
            
            # Avoid sampling neighbors of duplicate Peak nodes within the same subgraph.
            peak_node_list_non_repeat = sorted(set(peak_node_list), key = peak_node_list.index) 
            
            for index_peak, peak_node in enumerate(peak_node_list_non_repeat[ids:]):
                kmer_exp = Kmer_count_copy.transpose()[:, peak_node]
                motif_exp = Motif_count_copy.transpose()[:, peak_node]
                kmer_exp[kmer_exp < 3] = 0
                    
                kmer_node_list_temp = Subgraph_Segmentation(Kmer_count, peak_node, neighbor_node_num[2], 
                                                                                                np.squeeze(np.array(np.log(kmer_exp + 1))))
                motif_node_list_temp = Subgraph_Segmentation(Motif_count, peak_node, neighbor_node_num[3], 
                                                                                                np.squeeze(np.array(np.log(motif_exp + 1))))
                if kmer_node_list_temp == []:
                    peak_node_no_kmer_list.append(peak_node)
                if motif_node_list_temp == []:
                    peak_node_no_motif_list.append(peak_node)
                dic_peak[peak_node] = {"k": kmer_node_list_temp, "m": motif_node_list_temp}
                
                # Summarize the IDs of the neighboring K-mer nodes of all Peak nodes in the current subgraph.
                kmer_node_list = kmer_node_list + kmer_node_list_temp
                
                # Summarize the IDs of the neighboring Motif nodes of all Peak nodes in the current subgraph.
                motif_node_list = motif_node_list + motif_node_list_temp
                    
        cell_node_list = cell_node_id[i * cell_node_num:(i + 1) * cell_node_num]
        gene_node_list = sorted(set(gene_node_list))
        peak_node_list = sorted(set(peak_node_list))
        kmer_node_list = sorted(set(kmer_node_list))
        motif_node_list = sorted(set(motif_node_list))
        
        d = dict()
        d["cell_node_index"] = cell_node_list
        d["gene_node_index"] = gene_node_list
        d["peak_node_index"] = peak_node_list
        d["kmer_node_index"] = kmer_node_list
        d["motif_node_index"] = motif_node_list
        
        # Store subgraph structural information, 
        # namely the IDs of all Cell nodes, neighboring Gene nodes, neighboring Peak nodes, neighboring K-mer nodes, and neighboring Motif nodes.
        total_node_list.append(d)
        
    if cell_node_no_gene_list != []:
        print("Cell nodes " + str(sorted(set(cell_node_no_gene_list))) + " has no neighbor gene node can be selected.")
    if cell_node_no_peak_list != []:
        print("Cell nodes " + str(sorted(set(cell_node_no_peak_list))) + " has no neighbor peak node can be selected.")
    if peak_node_no_kmer_list != []:
        print("Peak nodes " + str(sorted(set(peak_node_no_kmer_list))) + " has no neighbor kmer node can be selected.")
    if peak_node_no_motif_list != []:
        print("Peak nodes " + str(sorted(set(peak_node_no_motif_list))) + " has no neighbor motif node can be selected.")
        
    return total_node_list, cell_node_id, dic_cell, dic_peak


# Normalize the weights of the edges in the graph.
def Norm_(edge_index, num_nodes, edge_weight = None, dtype = None):
    
    if edge_weight is None:
        edge_weight = torch.ones((edge_index.size(1), ),
                                                    dtype = dtype,
                                                    device = edge_index.device)
    edge_weight = edge_weight.view(-1)
    assert edge_weight.size(0) == edge_index.size(1)
    row, col = edge_index.detach()
    
    # Calculate the degree of each node in the graph.
    deg = scatter_add(edge_weight.clone(), row.clone(), dim = 0, dim_size = num_nodes)
    
    # Calculate the inverse square root of the degree matrix.
    deg_inv_sqrt = deg.pow(-1)
    deg_inv_sqrt[deg_inv_sqrt == float("inf")] = 0
        
    return deg_inv_sqrt, row, col


# Generate non-local graphs.
def Generate_Non_Local_Graph(args, feat_trans, H, A, num_edge, num_nodes):
    
    # Number of most relevant nodes for each node.
    K = args.K        
    x = F.relu(feat_trans(H))
    
    # Compute the similarity matrix between transformed node features.
    D_ = x@x.t()
    
    # Sort each row of the similarity matrix and return the indices of the top K most relevant nodes along with their similarity values.
    _, D_topk_indices = D_.t().sort(dim = 1, descending = True)
    D_topk_indices = D_topk_indices[:,:K]
    D_topk_value = D_.t()[torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K), D_topk_indices]
    
    # Edge index and edge weight of non-local graphs.
    edge_j = D_topk_indices.reshape(-1)
    edge_i = torch.arange(D_.shape[0]).unsqueeze(-1).expand(D_.shape[0], K).reshape(-1).to(H.device)
    edge_index = torch.stack([edge_i, edge_j])
    edge_value = (D_topk_value).reshape(-1)
    edge_value = D_topk_value.reshape(-1)
    
    return [edge_index, edge_value]


def Initial_Clustering(RNA_count, custom_n_neighbors = None, n_pcs = 40, custom_resolution = None, use_rep = None, batch = False, sample = None):

    print("\tWhen the number of cells is less than or equal to 500, it is recommended to set the resolution value to 0.2.")
    print("\tWhen the number of cells is within the range of 500 to 5000, the resolution value should be set to 0.5.")
    print("\tWhen the number of cells is greater than 5000, the resolution value should be set to 0.8.")

    adata = ad.AnnData(RNA_count.transpose(), dtype = "int32")
    
    # Use Segment_Function to obtain the parameters custom_resolution and custom_n_neighbors.
    if custom_resolution is None or custom_n_neighbors is None:
        resolution, n_neighbors = Segment_Function(adata.shape[0])
    else:
        resolution = custom_resolution
        n_neighbors = custom_n_neighbors

    sc.pp.normalize_total(adata, target_sum = 1e4)
    sc.pp.log1p(adata)
    sc.pp.scale(adata, max_value = 10)
    
    if batch is True:
        if sample is not None:
            adata.obs["batch"] = sample
            sc.pp.pca(adata)
            sce.pp.harmony_integrate(adata, "batch")
            sc.pp.neighbors(adata, n_neighbors = n_neighbors, n_pcs = n_pcs, use_rep = "X_pca_harmony")
        else:
            raise ValueError("The input must contain batch information.")
    else:
        # Construct a KNN graph based on the embeddings or n_pcs input by user.
        if use_rep is not None:
            adata.obsm["use_rep"] = use_rep
            sc.pp.neighbors(adata, use_rep = "use_rep", n_neighbors = n_neighbors)
        else:
            sc.pp.neighbors(adata, n_neighbors = n_neighbors, n_pcs = n_pcs)

    sc.tl.leiden(adata, resolution)
    return adata.obs["leiden"]


# Return cluster labels with a proportion below 3% and below 1%.
def Calculate_Frequency(lst):
    counter = Counter(lst)
    length = len(lst)
    low_frequency_elements = [element for element, count in counter.items() if count / length < 0.01]
    medium_frequency_elements = [element for element, count in counter.items() if count / length < 0.03 and count / length >= 0.01]
    rare_frequency_elements = [sorted(medium_frequency_elements), sorted(low_frequency_elements)]
    
    return rare_frequency_elements


# Return rare labels that account for less than 1% of the Counter object.
def Get_Rare_Items(counter_obj, threshold = 0.01):
    total = sum(counter_obj.values())
    sorted_items = sorted(counter_obj.items(), key = lambda x: x[1], reverse = True)
    rare_items = [item for item, count in sorted_items if count / total <= threshold and count > 20]
    
    return rare_items
    

def Calculate_Metrics(y_true, y_pred):
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    tpr = tp / (tp + fn)
    tnr = tn / (tn + fp)
    F1 = f1_score(y_true, y_pred)
    G_mean = np.sqrt(tpr * tnr)
    MCC = matthews_corrcoef(y_true, y_pred)
    Kappa = cohen_kappa_score(y_true, y_pred)

    return F1, G_mean, MCC, Kappa