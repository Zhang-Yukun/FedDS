from copy import deepcopy
import torch
import tqdm
from .base_selector import BaseSelector
from utils.logger import logger


class KCenterSelector(BaseSelector):

    def query(self, rd=0, n=0, tokenizer=None, use_model_path=None):
        # get all data embeddings using prev_rd's model
        embeddings_lst = self.get_embeddings_all_data(rd=rd, tokenizer=tokenizer, use_model_path=use_model_path)  # (num_all_data, emb_dim)
        embeddings = torch.stack(embeddings_lst)  # (num_all_data, hidden_dim=4096)
        device = embeddings.device
        embeddings = embeddings.float()  # fp32
        logger.info(f"*** Round {rd} **, **Embeddings.shape = {embeddings.shape}")
        dist_mat = torch.empty((0, embeddings.shape[0])).to(device)  # init empy dist_mat .shape=(0, num_all_data)
        # partion embeddings -> to compute distances
        step = 1000  # how many data in each partition? -> change this accordingly if you run into OOM on a single GPU
        for i in range(0, embeddings.shape[0], step):
            logger.info(f"*** Round {rd} **, **Dies_e -- Start Computing From Idx = {i}")
            e = embeddings[i:i + step, :]
            dist_e = torch.matmul(e, embeddings.t())
            torch.cuda.empty_cache()
            dist_mat = torch.concat([dist_mat, dist_e], dim=0)
            logger.info(f"*** Round {rd} **, **Computed dist_e.shape = {dist_e.shape}")

        sq = torch.tensor(dist_mat.diagonal(), device=device, requires_grad=False).reshape(len(self.labeled_idx),
                                                                                           1)  # diagonal ->  (num_all_data, 1)
        dist_mat *= -2
        dist_mat += sq
        dist_mat += sq.t()
        dist_mat = torch.sqrt(
            dist_mat)  # -> distances between all_data_embeddings (num_all_data, num_all_data) (diagonal=0)
        labeled_idxs_tmp = deepcopy(self.labeled_idx)
        mat = dist_mat[~labeled_idxs_tmp, :][:,
              labeled_idxs_tmp]  # unlabeled<->labeled distances (unlabeled_size, labeled_size)
        logger.info(f"*** Round {rd} **, **Computed dist_mat.shape = {mat.shape}")
        # sample n new datapoints from unlabeled_pool
        for i in tqdm.tqdm(range(n), ncols=n):
            mat_min = mat.min(dim=1).values  # (unlabeled_size, ) min_distance to existing labeled datapoint
            q_idx_tmp = mat_min.argmax()  # argmax(min_distance)
            q_idx = torch.arange(self.n_pool)[~labeled_idxs_tmp][q_idx_tmp]  # find its index in full dataset
            labeled_idxs_tmp[q_idx] = True  # add to labeled_pool
            left_unlabeled_idxs = torch.ones(mat.shape[0], dtype=torch.bool, device=device,
                                             requires_grad=False)  # BOOL: remaining unchosen data (in dist_mat)
            left_unlabeled_idxs[q_idx_tmp] = 0  #  newly added -> 0
            mat = mat[left_unlabeled_idxs, :].reshape(-1, mat.shape[
                1])  # left_unlabeled<->labeled distances (unlabeled_size-(i+1), labeled_size)
            mat = torch.concat((mat, dist_mat[~labeled_idxs_tmp, q_idx][:, None]),
                               dim=1)  # # left_unlabeled<->updated_labeled distances (unlabeled_size-(i+1), labeled_size+(i+1))
        new_idx = torch.arange(self.n_pool)[
            (self.labeled_idx ^ labeled_idxs_tmp)]  # new datapoints -> indices in full dataset
        logger.info(f"*** Round {rd} ** Added-Sample-idx = {new_idx}")
        self.update_rd(rd=rd, add_labeled_idx=new_idx)