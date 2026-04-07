import pandas as pd
import torch
import numpy as np
import h5py
from Bio import SeqIO


class PromoterDataset(torch.utils.data.Dataset):
    def __init__(self):
        self.shuffle = False
        self.seqs = []
        class ModelParameters:
            nt_embedding_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/inference_dataset/all_species_infer_dataset/lian_lab_aval_yeast_promoter_genera_signal_embeddings.h5'
            fasta_file = '/root/autodl-tmp/graph_model/Diffusion_directed_evolution/seqs_and_embeddings/inference_dataset/all_species_infer_dataset/lian_lab_aval_yeast_promoter_sequences.fasta'
            n_time_steps = 400

            random_order = False
            speed_balanced = True
            ncat = 4
            num_epochs = 200

            lr = 5e-4

        config = ModelParameters()
        with h5py.File(config.nt_embedding_file,'r') as h:
            self.seq_llm_embedding = torch.tensor(h['embeddings'][:], dtype=torch.float)

        for record in SeqIO.parse(config.fasta_file, 'fasta'):
            self.seqs.append(str(record.seq))
        
        #print(self.str_to_one_hot(self.seqs[0]))
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        #print(f"__getitem__ called with idx: {idx}")
        seq = self.str_to_one_hot(self.seqs[idx]) # L, 4
        #print(f'seq_shape:{seq.shape}')
        emb = self.seq_llm_embedding[idx] 
        return seq, emb
    
    def str_to_one_hot(self, seq):
        base_map = {'A':0, 'C':1, 'G':2, 'T':3}
        seq_tensor = torch.tensor([base_map[base] for base in seq], dtype=torch.long)
        seq_one_hot = torch.nn.functional.one_hot(seq_tensor, num_classes=4)
        return seq_one_hot

class CodonInfoConditionalGenerationDataset(torch.utils.data.Dataset):
    def __init__(self):
        
        species = 'sc'
        data_file_path = f'/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/data/{species}_dataset_with_promoters_500_and_codon_frequency.csv'
        data = pd.read_csv(data_file_path)
        data = data[data['TPM']>5].dropna(subset=['promoter_sequence', 'gene_sequence'])
        
        #sc_embeddings = np.load('/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/data/kazachstania_africana_cbs_2517_gca_000304475_promoter_embeddings_500bp.npy')
        
        # codon info
        codon_frequencies = data.iloc[:, -65:].values  
        # codon_count = data.iloc[:, -1:].values  
        # reshaped_frequencies = codon_frequencies.reshape(-1, 16, 4)
        # expanded_codon_count = np.expand_dims(codon_count, axis=-1)  
        # expanded_codon_count = np.tile(expanded_codon_count, (1,1,4))  
        # self.codon_data = torch.tensor(np.concatenate([reshaped_frequencies, expanded_codon_count], axis=1), dtype=torch.float)
        self.codon_data = torch.tensor(codon_frequencies, dtype=torch.float)
        #self.embeddings = torch.tensor(sc_embeddings, dtype=torch.float)
        
        self.signal = torch.cat([self.codon_data, self.embeddings], dim=1)
        
        # promoter seqs
        self.seqs = data['promoter_sequence']
        self.seqs = torch.stack([self.str_to_one_hot(seq) for seq in self.seqs])
        
        print(self.signal.shape)
        print(self.seqs.shape)
        
        #self.seqs = torch.cat([self.seqs, torch.tensor(self.codon_data, dtype=torch.float32)], dim=1)
        
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx] # L, 4
        # codon_info = self.codon_data[idx]
        signal = self.signal[idx]
        return seq, signal
    
    def str_to_one_hot(self, seq):
        base_map = {'A':0, 'C':1, 'G':2, 'T':3}
        seq_tensor = torch.tensor([base_map[base] for base in seq], dtype=torch.long)
        seq_one_hot = torch.nn.functional.one_hot(seq_tensor, num_classes=4)
        return seq_one_hot
    
class SpeciesAwareCodonInfoCFGDataset(torch.utils.data.Dataset):
    def __init__(self):
        
        data_file_path = f'/root/autodl-tmp/graph_model/Diffusion_directed_evolution/SpeciesLM/cross_species_data/lianlab_aval_yeast_promoter_gene_info.csv'
        data = pd.read_csv(data_file_path)
        data = data.dropna(subset=['promoter_sequence', 'gene_sequence','AAA', 'AAC', 'AAG', 'AAT', 'ACA',
                                        'ACC', 'ACG', 'ACT', 'AGA', 'AGC', 'AGG', 'AGT', 'ATA', 'ATC', 'ATG',
                                        'ATT', 'CAA', 'CAC', 'CAG', 'CAT', 'CCA', 'CCC', 'CCG', 'CCT', 'CGA',
                                        'CGC', 'CGG', 'CGT', 'CTA', 'CTC', 'CTG', 'CTT', 'GAA', 'GAC', 'GAG',
                                        'GAT', 'GCA', 'GCC', 'GCG', 'GCT', 'GGA', 'GGC', 'GGG', 'GGT', 'GTA',
                                        'GTC', 'GTG', 'GTT', 'TAA', 'TAC', 'TAG', 'TAT', 'TCA', 'TCC', 'TCG',
                                        'TCT', 'TGA', 'TGC', 'TGG', 'TGT', 'TTA', 'TTC', 'TTG', 'TTT',])
        
        data = data[data['promoter_sequence'].str.len() == 500].reset_index(drop=True)
        
        codon_frequencies = data.iloc[:, -65:-1].values 
        self.codon_data = torch.tensor(codon_frequencies, dtype=torch.float)
        
        species_unique = sorted(data["species"].unique())
        self._sp2idx = {sp: i for i, sp in enumerate(species_unique)}
        species_idx = data["species"].map(self._sp2idx).to_numpy(dtype="int64")
        self.species = torch.nn.functional.one_hot(
            torch.tensor(species_idx, dtype=torch.long),
            num_classes=len(species_unique)
        ).to(torch.float32)  # [N, N_species]
        
        self.seqs = data['promoter_sequence']
        self.seqs = torch.stack([self.str_to_one_hot(seq) for seq in self.seqs])
        print(self.seqs.shape)
        
    def __len__(self):
        return len(self.seqs)

    def __getitem__(self, idx):
        seq = self.seqs[idx] # L, 4
        codon_info = self.codon_data[idx]
        species = self.species[idx]
        return seq, codon_info, species
    
    def str_to_one_hot(self, seq):
        base_map = {'A':0, 'C':1, 'G':2, 'T':3}
        seq_tensor = torch.tensor([base_map[base] for base in seq], dtype=torch.long)
        seq_one_hot = torch.nn.functional.one_hot(seq_tensor, num_classes=4)
        return seq_one_hot
