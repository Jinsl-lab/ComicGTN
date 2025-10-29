import math
import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm
from scipy.sparse import csr_matrix
from torch_geometric.utils import add_self_loops
from .settings import *
from .utils import *
from .GCNConv import *
from .FastGTNConv import *


device = f"cuda" if torch.cuda.is_available() else "cpu"


class FastGTNs(nn.Module):
    def __init__(self, num_edge_type, w_in_list, num_nodes, args = None):
        super(FastGTNs, self).__init__()
        self.args = args
        self.num_nodes = num_nodes
        self.num_FastGTN_layers = args.num_FastGTN_layers
        
        # Define a linear projection layer for each node type to standardize input features to 512 dimensions.
        self.feature_projections = nn.ModuleList([nn.Linear(w_in, 512) for w_in in w_in_list])
        fastGTNs = []
        
        for i in range(args.num_FastGTN_layers):
            if i == 0:
                fastGTNs.append(FastGTN(num_edge_type, 512, num_nodes, args))
            else:
                fastGTNs.append(FastGTN(num_edge_type, args.node_dim, num_nodes, args))
                
        self.fastGTNs = nn.ModuleList(fastGTNs)

    def forward(self, A, X_list, num_nodes = None, epoch = None):
        if num_nodes is None:
            num_nodes = self.num_nodes
            
        X = [proj(x) for proj, x in zip(self.feature_projections, X_list)]
        
        # After vertically concatenating node features with unified dimensions, return the updated node features and weight matrix list.
        X = torch.cat(X, dim = 0)
        H_, Ws = self.fastGTNs[0](A, X, num_nodes = num_nodes, epoch = epoch)
        
        for i in range(1, self.num_FastGTN_layers):
            H_, Ws = self.fastGTNs[i](A, H_, num_nodes = num_nodes)
            
        return H_, Ws
    
    
def Prepare_Batch_Data(RNA_count, ATAC_count, Kmer_count, Motif_count, total_node_idx, batch_id, device):
    cell_node_index = total_node_idx[batch_id]["cell_node_index"]
    gene_node_index = total_node_idx[batch_id]["gene_node_index"]
    peak_node_index = total_node_idx[batch_id]["peak_node_index"]
    kmer_node_index = total_node_idx[batch_id]["kmer_node_index"]
    motif_node_index = total_node_idx[batch_id]["motif_node_index"]

    cell_node_feature = torch.tensor(RNA_count[:, list(cell_node_index)].T.toarray(), dtype = torch.float32).to(device)
    gene_node_feature = torch.tensor(RNA_count[list(gene_node_index), ].toarray(), dtype = torch.float32).to(device)
    peak_node_feature = torch.tensor(ATAC_count[list(peak_node_index), ].toarray(), dtype = torch.float32).to(device)
    kmer_node_feature = torch.tensor(Kmer_count[:, list(kmer_node_index), ].T, dtype = torch.float32).to(device)
    motif_node_feature = torch.tensor(Motif_count[:, list(motif_node_index), ].T, dtype = torch.float32).to(device)
    total_node_feature = [cell_node_feature, gene_node_feature, peak_node_feature, kmer_node_feature, motif_node_feature]

    node_type = np.array(list(np.zeros(len(cell_node_index))) + list(np.ones(len(gene_node_index))) +
                                        list(np.ones(len(peak_node_index)) * 2) + list(np.ones(len(kmer_node_index)) * 3) +
                                        list(np.ones(len(motif_node_index)) * 4))
    node_type = torch.tensor(node_type, dtype = torch.long).to(device)
    dim = len(node_type)

    gene_cell_sub = RNA_count[list(gene_node_index), ][:, list(cell_node_index)]
    A_GC = csr_matrix((np.ones(gene_cell_sub.nnz), (list(np.nonzero(gene_cell_sub)[0] + gene_cell_sub.shape[1]),
                                    list(np.nonzero(gene_cell_sub)[1]))), shape = (dim, dim))
    A_CG = A_GC.transpose()
    peak_cell_sub = ATAC_count[list(peak_node_index), ][:, list(cell_node_index)]
    A_PC = csr_matrix((np.ones(peak_cell_sub.nnz), (list(np.nonzero(peak_cell_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]), 
                                    list(np.nonzero(peak_cell_sub)[1]))), shape = (dim, dim))
    A_CP = A_PC.transpose()
    peak_kmer_sub = csr_matrix(Kmer_count[list(peak_node_index), ][:, list(kmer_node_index)])
    A_KP = csr_matrix((np.ones(peak_kmer_sub.nnz), (list(np.nonzero(peak_kmer_sub)[1] + gene_cell_sub.shape[0] +
                                   gene_cell_sub.shape[1] + peak_cell_sub.shape[0]), 
                                   list(np.nonzero(peak_kmer_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]))), shape = (dim, dim))
    A_PK = A_KP.transpose()
    peak_motif_sub = csr_matrix(Motif_count[list(peak_node_index), ][:, list(motif_node_index)])
    A_MP = csr_matrix((np.ones(peak_motif_sub.nnz), (list(np.nonzero(peak_motif_sub)[1] + gene_cell_sub.shape[0] +
                                    gene_cell_sub.shape[1] + peak_cell_sub.shape[0] + peak_kmer_sub.shape[1]),
                                    list(np.nonzero(peak_motif_sub)[0] + gene_cell_sub.shape[0] + gene_cell_sub.shape[1]))), shape = (dim, dim))
    A_PM = A_MP.transpose()
    A = []
    edge_type = [A_CG, A_CP, A_PK, A_PM]
    num_nodes = edge_type[0].shape[0]
    
    for i, edge in enumerate(edge_type):
        edge_tmp = torch.from_numpy(np.vstack((edge.nonzero()[1], edge.nonzero()[0]))).long().to(device)
        value_tmp = torch.ones(edge_tmp.shape[1]).float().to(device)
        edge_tmp, value_tmp = add_self_loops(edge_tmp, edge_attr = value_tmp, fill_value = 1e-20, num_nodes = num_nodes)
        deg_inv_sqrt, deg_row, deg_col = Norm_(edge_tmp.detach(), num_nodes, value_tmp.detach())
        value_tmp = deg_inv_sqrt[deg_row] * value_tmp
        A.append((edge_tmp, value_tmp))
    edge_tmp = torch.stack((torch.arange(0, num_nodes), torch.arange(0, num_nodes))).long().to(device)
    value_tmp = torch.ones(num_nodes).float().to(device)
    A.append((edge_tmp, value_tmp))

    return total_node_feature, node_type, A, gene_cell_sub, peak_cell_sub, peak_kmer_sub, peak_motif_sub


class NodeFeatureEmbedding(nn.Module):
    def __init__(self, RNA_count, ATAC_count, Kmer_count, Motif_count, total_node_idx, ini_clu,
                       rare_labels, args, device, num_edge_type, epochs = 1):
        super(NodeFeatureEmbedding, self).__init__()
        self.RNA_count = RNA_count
        self.ATAC_count = ATAC_count
        self.Kmer_count = Kmer_count
        self.Motif_count = Motif_count
        self.total_node_idx = total_node_idx
        self.ini_clu = ini_clu
        self.rare_labels = rare_labels
        self.args = args
        self.device = device
        self.num_edge_type = num_edge_type
        self.epochs = epochs
        self.w_in_list = [RNA_count.shape[0], RNA_count.shape[1], ATAC_count.shape[1], Kmer_count.shape[0], Motif_count.shape[0]]

        self.GTN = FastGTNs(num_edge_type = self.num_edge_type,
                                          w_in_list = self.w_in_list,
                                          num_nodes = None,
                                          args = self.args).to(self.device)

        self.MuRaL = MultiLevelRareLoss(rare_labels = self.rare_labels, 
                                                              rare_weight = args.rare_weight, 
                                                              smoothing = args.smoothing)
        
        self.SuCoL = ImprovedSupConLoss(temperature = args.temperature,
                                                                 hard_neg_k = args.hard_neg_k,
                                                                 rand_neg_ratio = args.rand_neg_ratio)
    
        self.optimizer = torch.optim.AdamW(self.GTN.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, "min", factor = 0.5, 
                                                                                                            patience = 5, verbose = True)

    def train_process(self, batch_num):
        print("Node feature dimensionality reduction training is being performed, which may take a while.")

        for epoch in range(self.epochs):
            for batch_id in tqdm(np.arange(batch_num)):
                total_node_feature, node_type, A, gene_cell_sub, peak_cell_sub, peak_kmer_sub, peak_motif_sub = Prepare_Batch_Data(
                    self.RNA_count, self.ATAC_count, self.Kmer_count, self.Motif_count, 
                    self.total_node_idx, batch_id, self.device)
                H_, Ws = self.GTN.forward(A, total_node_feature, num_nodes = len(node_type))
                H_ = H_.to(self.device)
                cell_node_emb = H_[node_type == 0]
                gene_node_emb = H_[node_type == 1]
                peak_node_emb = H_[node_type == 2]
                kmer_node_emb = H_[node_type == 3]
                motif_node_emb = H_[node_type == 4]
                cluster_labels = torch.LongTensor(np.array(self.ini_clu)[self.total_node_idx[batch_id]["cell_node_index"]]).to(self.device)

                loss_MuRaL = self.MuRaL(cell_node_emb, cluster_labels)
                print(loss_MuRaL)
                loss_SuCoL = self.SuCoL(cell_node_emb, cluster_labels)
                print(loss_SuCoL)
                loss_cluster = loss_MuRaL + loss_SuCoL
                print(loss_cluster)

                loss_Cos = 0
                g = [int(i) for i in cluster_labels]
                for i in set([int(k) for k in cluster_labels]):
                    h = cell_node_emb[[True if i == j else False for j in g]]
                    ll = F.cosine_similarity(h[list(range(h.shape[0])) * h.shape[0],],
                                                        h[[v for v in range(h.shape[0]) for i in range(h.shape[0])]]).mean()
                    loss_Cos = ll + loss_Cos
                
                print(loss_Cos)
                loss = loss_cluster - loss_Cos
                print(loss)

                self.optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.GTN.parameters(), max_norm = 1.0)
                self.optimizer.step()

        print("Node feature dimensionality reduction training has been completed.")
        return self.GTN, cell_node_emb, gene_node_emb, peak_node_emb, kmer_node_emb, motif_node_emb

    
class Comic(nn.Module):
    def __init__(self, GTN, batch_num, rare_labels, args, device, epochs = 1):
        super(Comic, self).__init__()
        self.GTN = GTN
        self.batch_num = batch_num
        self.rare_labels = rare_labels
        self.args = args
        self.device = device
        self.epochs = epochs
        self.GTN_optimizer = torch.optim.AdamW(self.GTN.parameters(), lr = args.lr, weight_decay = args.weight_decay)
        self.MuRaL = MultiLevelRareLoss(rare_labels = self.rare_labels, 
                                                              rare_weight = args.rare_weight, 
                                                              smoothing = args.smoothing)
        
    def forward(self, total_node_idx, RNA_count, ATAC_count, Kmer_count, Motif_count, ini_clu):
        ini_clu = np.array(ini_clu)

        for epoch in range(self.epochs):
            for batch_id in tqdm(np.arange(self.batch_num)):
                total_node_feature, node_type, A, gene_cell_sub, peak_cell_sub, peak_kmer_sub, peak_motif_sub = Prepare_Batch_Data(
                    RNA_count, ATAC_count, Kmer_count, Motif_count, 
                    total_node_idx, batch_id, self.device)

                H_, Ws = self.GTN.forward(A, total_node_feature, num_nodes = len(node_type))
                H_ = H_.to(self.device)
                cell_node_emb = H_[node_type == 0]
                gene_node_emb = H_[node_type == 1]
                peak_node_emb = H_[node_type == 2]
                kmer_node_emb = H_[node_type == 3]
                motif_node_emb = H_[node_type == 4]

                decoder1 = torch.mm(gene_node_emb, cell_node_emb.t())
                decoder2 = torch.mm(peak_node_emb, cell_node_emb.t())
                decoder3 = torch.mm(peak_node_emb, kmer_node_emb.t())
                decoder4 = torch.mm(peak_node_emb, motif_node_emb.t())
                
                gene_cell_sub = torch.tensor(gene_cell_sub.toarray(), dtype = torch.float32).to(self.device)
                peak_cell_sub = torch.tensor(peak_cell_sub.toarray(), dtype = torch.float32).to(self.device)
                peak_kmer_sub = torch.tensor(peak_kmer_sub.toarray(), dtype = torch.float32).to(self.device)
                peak_motif_sub = torch.tensor(peak_motif_sub.toarray(), dtype = torch.float32).to(self.device)
                
                logp_x1 = F.log_softmax(decoder1, dim = -1)
                p_y1 = F.softmax(gene_cell_sub, dim = -1)
                
                logp_x2 = F.log_softmax(decoder2, dim = -1)
                p_y2 = F.softmax(peak_cell_sub, dim = -1)
                
                logp_x3 = F.log_softmax(decoder3, dim = -1)
                p_y3 = F.softmax(peak_kmer_sub, dim = -1)
                
                logp_x4 = F.log_softmax(decoder4, dim = -1)
                p_y4 = F.softmax(peak_motif_sub, dim = -1)

                KL1 = F.kl_div(logp_x1, p_y1, reduction = "batchmean")
                print(KL1)
                KL2 = F.kl_div(logp_x2, p_y2, reduction = "batchmean")
                print(KL2)
                KL3 = F.kl_div(logp_x3, p_y3, reduction = "batchmean")
                print(KL3)
                KL4 = F.kl_div(logp_x4, p_y4, reduction = "batchmean")
                print(KL4)
    
                loss_KL_I = KL1 + KL2 + KL3 + KL4
                print(loss_KL_I)

                cluster_labels = torch.LongTensor(ini_clu[total_node_idx[batch_id]["cell_node_index"]]).to(self.device)
                loss_MuRaL = self.MuRaL(cell_node_emb, cluster_labels)
                print(loss_MuRaL)
                
                loss = loss_KL_I + loss_MuRaL
                print(loss)

                self.GTN_optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.GTN.parameters(), max_norm = 1.0)
                self.GTN_optimizer.step()

        return self.GTN

    def train_process(self, total_node_idx, RNA_count, ATAC_count, Kmer_count, Motif_count, ini_clu):
        self.train()
        print("ComicGTN model training is being performed, which may take a while.")
        ComicGTN = self.forward(total_node_idx, RNA_count, ATAC_count, Kmer_count, Motif_count, ini_clu)
        print("The training for the ComicGTN model has been completed.")
        return ComicGTN
    

def ComicGTN_test(RNA_count, ATAC_count, Kmer_count, Motif_count, total_node_idx, cell_node_idx, 
                                cell_node_num, ComicGTN, device, co_emb = True):
    batch_num = math.ceil(len(cell_node_idx) / cell_node_num)
    cell_embeddings = []
    gene_embeddings = {}
    peak_embeddings = {}
    kmer_embeddings = {}
    motif_embeddings = {}
    cluster_label_pred = []

    with torch.no_grad():
        for batch_id in tqdm(range(batch_num)):
            total_node_feature, node_type, A, gene_cell_sub, peak_cell_sub, peak_kmer_sub, peak_motif_sub = Prepare_Batch_Data(
                RNA_count, ATAC_count, Kmer_count, Motif_count, 
                total_node_idx, batch_id, device)

            node_emb, Ws = ComicGTN.forward(A, total_node_feature, num_nodes = len(node_type))
            node_emb = node_emb.to(device)
            cell_node_emb = node_emb[node_type == 0]
            gene_node_emb = node_emb[node_type == 1]
            peak_node_emb = node_emb[node_type == 2]
            kmer_node_emb = node_emb[node_type == 3]
            motif_node_emb = node_emb[node_type == 4]

            if device == "cuda":
                cell_node_emb = cell_node_emb.cpu()
            cell_embeddings.append(cell_node_emb.detach().numpy())

            cluster_pred = list(cell_node_emb.argmax(dim = 1).detach().numpy())
            cluster_label_pred.extend(cluster_pred)

            for i, idx in enumerate(total_node_idx[batch_id]["gene_node_index"]):
                if idx not in gene_embeddings:
                    gene_embeddings[idx] = []
                gene_embeddings[idx].append(gene_node_emb[i].cpu().detach().numpy())

            for i, idx in enumerate(total_node_idx[batch_id]["peak_node_index"]):
                if idx not in peak_embeddings:
                    peak_embeddings[idx] = []
                peak_embeddings[idx].append(peak_node_emb[i].cpu().detach().numpy())

            for i, idx in enumerate(total_node_idx[batch_id]["kmer_node_index"]):
                if idx not in kmer_embeddings:
                    kmer_embeddings[idx] = []
                kmer_embeddings[idx].append(kmer_node_emb[i].cpu().detach().numpy())

            for i, idx in enumerate(total_node_idx[batch_id]["motif_node_index"]):
                if idx not in motif_embeddings:
                    motif_embeddings[idx] = []
                motif_embeddings[idx].append(motif_node_emb[i].cpu().detach().numpy())

    cell_embeddings = np.vstack(cell_embeddings)
    cluster_label_pred = np.array(cluster_label_pred)

    if co_emb:
        gene_embedding_avg = np.zeros((RNA_count.shape[0], gene_node_emb.shape[1]))
        peak_embedding_avg = np.zeros((ATAC_count.shape[0], peak_node_emb.shape[1]))
        kmer_embedding_avg = np.zeros((Kmer_count.shape[1], kmer_node_emb.shape[1]))
        motif_embedding_avg = np.zeros((Motif_count.shape[1], motif_node_emb.shape[1]))

        for idx, emb_list in gene_embeddings.items():
            gene_embedding_avg[idx] = np.mean(emb_list, axis = 0)

        for idx, emb_list in peak_embeddings.items():
            peak_embedding_avg[idx] = np.mean(emb_list, axis = 0)

        for idx, emb_list in kmer_embeddings.items():
            kmer_embedding_avg[idx] = np.mean(emb_list, axis = 0)

        for idx, emb_list in motif_embeddings.items():
            motif_embedding_avg[idx] = np.mean(emb_list, axis = 0)

        ComicGTN_result = {"predicted_cluster_label": cluster_label_pred,
                                         "cell_node_embedding": cell_embeddings,
                                         "gene_node_embedding": gene_embedding_avg,
                                         "peak_node_embedding": peak_embedding_avg,
                                         "kmer_node_embedding": kmer_embedding_avg,
                                         "motif_node_embedding": motif_embedding_avg}

        return ComicGTN_result
    
    else:
        ComicGTN_result = {"predicted_cluster_label": cluster_label_pred, "cell_node_embedding": cell_embeddings}
        return ComicGTN_result