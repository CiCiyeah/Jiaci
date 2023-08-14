import numpy as np
import networkx as nx
from glob import glob
import pandas as pd
from tqdm import tqdm
import subprocess

import signal
from contextlib import contextmanager

class TimeoutException(Exception): pass

@contextmanager
def time_limit(seconds):
    def signal_handler(signum, frame):
        raise TimeoutException("Timed out!")
    signal.signal(signal.SIGALRM, signal_handler)
    signal.alarm(seconds)
    try:
        yield
    finally:
        signal.alarm(0)

class BasePairProbabilities:
    def __init__(self, file_name):
        self.seq_len = 10
        self.min_len = 10
        self.seq_name = file_name.split(".")[-2].split("/")[-1]
        self.bpp_mat = np.zeros((self.seq_len, self.seq_len))
        if file_name.endswith(".npy"):
            self.read_npyfile(file_name)
        elif file_name.endswith(".ps") or file_name.endswith(".eps"):
            self.read_dotfile(file_name)
        elif file_name.endswith(".bpp"):
            self.read_bppfile(file_name)
        self.bps_mat = self.calculate_base_pair_score() # base pair scores

    def read_dotfile(self, dotfile):
        seq = ''
        seqflag = 0
        bpp = []
        with open(dotfile, 'r') as f:
            while True:
                line = f.readline()
                if not line:
                    break
                # read the input sequence
                if seqflag == 1 and "def" in line:
                    seqflag += 1
                if seqflag == 1:
                    seq += line
                    seq = seq.replace("\\", "")
                    seq = "".join(seq.split())
                if line.startswith("/sequence"):
                    seqflag = 1

                # read ensemble base pair probabilities
                if line.strip().endswith("ubox") and (not "sqrt" in line):
                    bpp.append(line)

        if self.seq_len != len(seq):
            self.seq_len = len(seq)
            self.bpp_mat = np.zeros((self.seq_len, self.seq_len))
        
        for d in bpp:
            s = d.split()
            i = int(s[0]) - 1
            j = int(s[1]) - 1
            self.bpp_mat[i, j] = float(s[2]) * float(s[2])
            self.bpp_mat[j, i] = self.bpp_mat[i, j]
    
    def read_npyfile(self, npyfile):
        tmp_mat = np.load(npyfile)
        if tmp_mat.shape != (self.seq_len, self.seq_len):
            self.seq_len = tmp_mat.shape[0]
        self.bpp_mat = tmp_mat

    def read_bppfile(self, bppfile):
        mat = []
        with open(bppfile, 'r') as f:
            for line in f.readlines():
                row = line.split()
                row = [float(elem) for elem in row]
                mat.append(row)
        mat = np.array(mat)
        assert mat.shape[0] == mat.shape[1]
        self.seq_len = mat.shape[0]
        self.bpp_mat = mat

    def calculate_base_pair_score(self, pnull=0.0005, eps=1e-10):
        eps_mat = np.ones_like(self.bpp_mat) * eps
        tmp = np.where(self.bpp_mat > eps_mat, self.bpp_mat, eps_mat)
        tmp = np.log(tmp / pnull)
        scores = np.where(tmp > eps_mat, tmp, eps_mat) / np.log(1 / pnull)

        return scores  

    def calculate_louvain_communities(self, pnull=0.25, neighbour_wt=0.2, resolution=0.01):
        mat = self.bpp_mat
        # import ipdb; ipdb.set_trace()
        mat[mat < pnull] = 0
        # mat += np.eye(self.seq_len)
        mat += np.eye(self.seq_len, k=-1) * neighbour_wt
        mat += np.eye(self.seq_len, k=1) * neighbour_wt
        G = nx.from_numpy_array(mat)
        res = nx.community.louvain_communities(G, resolution=resolution)
        return res
    
    def calculate_fitness_function(self, k, l, flanking=10, eps=1e-10):
        if k < flanking or k >= self.seq_len - self.min_len - flanking or l < k + self.min_len - 1 or l >= self.seq_len - flanking:
            return 0

        I_k = np.sum(self.bps_mat[k: l + 1, k: l + 1], axis=1)
        O_k = np.sum(self.bps_mat[k: l + 1, :], axis=1) - I_k
        I_kl = np.log(np.where(I_k < np.ones_like(I_k), I_k, np.ones_like(I_k))).sum()
        O_kl = np.log(1 - np.where(O_k < (1 - eps) * np.ones_like(O_k), O_k, (1 - eps) * np.ones_like(O_k))).sum()
        S_k = np.sum(self.bps_mat[k  - flanking: k, l + 1: l + flanking], axis=1)
        S_kl = np.log(1 - np.where(S_k < (1 - eps) * np.ones_like(S_k), S_k, (1 - eps) * np.ones_like(S_k))).sum()
        
        len_kl = l - k + 1
        D_kl_w = I_kl / len_kl + O_kl / len_kl + S_kl / flanking

        return np.exp(D_kl_w)
    
    def generate_domain_candidates(self):
        while True:
            try:
                with time_limit(10):
                    subgraphs = self.calculate_louvain_communities()
                    break
            except TimeoutException as e:
                print("Time out, try again")

        subgraphs = [np.array(sorted(list(subgraph))) for subgraph in subgraphs]
        # new_subgraphs = []
        # for subgraph in subgraphs:
        #     for i, j in zip(subgraph, subgraph[1:]):
        #         if j - i > 1:
        #             new_subgraphs.append(subgraph[:int(np.where(subgraph == j)[0])])
        #             subgraph = subgraph[int(np.where(subgraph == j)[0]):]
        #     new_subgraphs.append(subgraph)
        #
        # critical_positions = set()
        # for new_subgraph in new_subgraphs:
        #     critical_positions.add(new_subgraph[0])
        #     critical_positions.add(new_subgraph[-1] + 1)
        #
        # critical_positions = sorted(list(critical_positions))
        #
        # domain_candidates = []
        #
        # for k in range(len(critical_positions)):
        #     for l in range(k + 1, len(critical_positions)):
        #         domain_candidates.append((critical_positions[k], critical_positions[l] - 1))
        domain_candidates = []
        for subgraph in subgraphs:
            domain_candidates.append((subgraph[0], subgraph[-1] + 1))   # consistent with Rfam

        return domain_candidates

    # def generate_domain_candidates(self):
    #     pred_dict = parse_rnalfold_results("./data/rnacentral/results.rnalfold")
    #     critical_points = set()
    #     for pairs in pred_dict[self.seq_name]:
    #         critical_points.add(pairs[0])
    #         critical_points.add(pairs[1])
    #     domain_candidates = []
    #     critical_points = list(critical_points)
    #     for k in range(len(critical_points)):
    #         for l in range(k + 1, len(critical_points)):
    #             domain_candidates.append((critical_points[k], critical_points[l] - 1))
    #     return domain_candidates


    def predict_domains(self):
        domain_candidates = self.generate_domain_candidates()
        # domain_scores = []
        #
        # for k, l in domain_candidates:
        #     domain_scores.append(self.calculate_fitness_function(k, l))
        #
        # domain_results = {domain_candidates[i]: domain_scores[i] for i in range(len(domain_candidates))}
        return domain_candidates
                        
    
if __name__ == "__main__":

    input_bpp_dir = "/data2/hongliang/rna_split/bpp_res/7/"
    input_csv_path = "/data2/hongliang/rna_split/rnacentral_short/rnacentral_fltd_shrt_low_dom_prop_7.fasta"
    output_fasta_path = "/data2/hongliang/rna_split/split_res/7.fasta"
    output_csv_path = "/data2/hongliang/rna_split/split_res/7.csv"

    file_names = glob(input_bpp_dir + "/*.bpp")
    df = pd.read_csv(input_csv_path, sep='\n', lineterminator='>', header=None, index_col=False, names=['seq_name', 'seq', 'ph'])
    df = df.drop_duplicates(subset=['seq_name'], ignore_index=True)
    # df = pd.read_csv(input_csv_path)

    df = df.set_index('seq_name')

    #out_df = pd.DataFrame(["seq_name, seq_start, seq_stop"])
    out_name = []
    out_start = []
    out_stop = []
    out_fasta_file = open(output_fasta_path, "w")

    # import ipdb; ipdb.set_trace()
    for i, file_name in enumerate(tqdm(file_names)):
        bpp = BasePairProbabilities(file_name)
        res = bpp.predict_domains()
        seq_name = file_name.split(".")[-2].split("/")[-1]
        seq = df.loc[seq_name, 'seq']

        for seq_start, seq_stop in res:
            # save to fasta
            out_fasta_file.write(">" + seq_name + "/" + str(seq_start) + "-" + str(seq_stop) + "\n")
            out_fasta_file.write(str(seq[seq_start: seq_stop]) + "\n")
            # save to dataframe
            # out_df = pd.concat([out_df, pd.DataFrame([seq_name, seq_start, seq_stop])])
            out_name.append(seq_name)
            out_start.append(seq_start)
            out_stop.append(seq_stop)

        break

    out_fasta_file.close()
    out_df = pd.DataFrame({"seq_name": out_name, "seq_start": out_start, "seq_stop": out_stop})
    out_df.to_csv(output_csv_path, index=False)


        # seq_start = df.loc[seq_name, 'seq_start'].astype(int)
        # seq_stop = df.loc[seq_name, 'seq_stop'].astype(int)
        # seq_start, seq_stop = min(seq_start, seq_stop), max(seq_start, seq_stop)
        # seq_len = df.loc[seq_name, 'seq_len'].astype(int)

        # result = sorted(res.items(), key=lambda s: s[1], reverse=True)[0][0]
        # df.loc[seq_name, 'seq'] = df.loc[seq_name, 'seq'][result[0]: result[1]]
        # if seq_stop < result[0] or result[1] < seq_start:
        #     ious.append(0)
        #     proportions.append((result[1] - result[0]) / (seq_stop - seq_start))
        #     coverage.append(0)
        # else:
        #     inter_start = max(seq_start, result[0])
        #     inter_end = min(seq_stop, result[1])
        #     union_start = min(seq_start, result[0])
        #     union_end = max(seq_stop, result[1])
        #     ious.append((inter_end - inter_start + 1) / (union_end - union_start + 1))
        #     proportions.append((result[1] - result[0] + 1) / (seq_stop - seq_start + 1))
        #     coverage.append((inter_end - inter_start + 1) / (seq_stop - seq_start + 1))
        # cut.append((result[1] - result[0] + 1) / seq_len)


        # find the result with highest iou
        # results = sorted(res.items(), key=lambda s: s[1], reverse=True)
        #
        # highest_iou = 0
        # selected_result = results[0][0]
        #
        # for result_pair in results:
        #     result = result_pair[0]
        #     if seq_stop < result[0] or result[1] < seq_start:
        #         continue
        #     else:
        #         inter_start = max(seq_start, result[0])
        #         inter_end = min(seq_stop, result[1])
        #         union_start = min(seq_start, result[0])
        #         union_end = max(seq_stop, result[1])
        #         iou = (inter_end - inter_start + 1) / (union_end - union_start + 1)
        #         if iou > highest_iou:
        #             highest_iou = iou
        #             selected_result = result
        #
        # if seq_stop < selected_result[0] or selected_result[1] < seq_start:
        #     ious.append(0)
        #     proportions.append((selected_result[1] - selected_result[0]) / (seq_stop - seq_start))
        #     coverage.append(0)
        # else:
        #     inter_start = max(seq_start, selected_result[0])
        #     inter_end = min(seq_stop, selected_result[1])
        #     union_start = min(seq_start, selected_result[0])
        #     union_end = max(seq_stop, selected_result[1])
        #     ious.append((inter_end - inter_start + 1) / (union_end - union_start + 1))
        #     proportions.append((selected_result[1] - selected_result[0] + 1) / (seq_stop - seq_start + 1))
        #     coverage.append((inter_end - inter_start + 1) / (seq_stop - seq_start + 1))
        # cut.append((selected_result[1] - selected_result[0] + 1) / seq_len)
        # df.loc[seq_name, 'seq'] = df.loc[seq_name, 'seq'][max(0, seq_start - int(0.3 * seq_len)): min(seq_stop + int(0.3 * seq_len), seq_len)]
        # df.loc[seq_name, 'seq'] = df.loc[seq_name, 'seq'][seq_start: seq_stop]
        # df.loc[seq_name, 'seq'] = df.loc[seq_name, 'seq'][selected_result[0]: selected_result[1]]

    # ious = np.array(ious)
    # proportions = np.array(proportions)
    # coverage = np.array(coverage)
    # cut = np.array(cut)
    # import ipdb; ipdb.set_trace()
    # print(ious)
    # print(ious.mean())
    # print(proportions.mean())
    # print(coverage.mean())
    # print(cut.mean())
    # df.to_csv("./data/rnacentral/rnacentral_fltd_shrt_low_domain_prop_sampled_seg_graph_part.csv")
    

