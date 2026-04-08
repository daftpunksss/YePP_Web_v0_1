from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import pandas as pd
import torch

from predictor_web import config


# Keep exact cfg_usage ordering.
CODON_COLS = [
    "AAA",
    "AAC",
    "AAG",
    "AAT",
    "ACA",
    "ACC",
    "ACG",
    "ACT",
    "AGA",
    "AGC",
    "AGG",
    "AGT",
    "ATA",
    "ATC",
    "ATG",
    "ATT",
    "CAA",
    "CAC",
    "CAG",
    "CAT",
    "CCA",
    "CCC",
    "CCG",
    "CCT",
    "CGA",
    "CGC",
    "CGG",
    "CGT",
    "CTA",
    "CTC",
    "CTG",
    "CTT",
    "GAA",
    "GAC",
    "GAG",
    "GAT",
    "GCA",
    "GCC",
    "GCG",
    "GCT",
    "GGA",
    "GGC",
    "GGG",
    "GGT",
    "GTA",
    "GTC",
    "GTG",
    "GTT",
    "TAA",
    "TAC",
    "TAG",
    "TAT",
    "TCA",
    "TCC",
    "TCG",
    "TCT",
    "TGA",
    "TGC",
    "TGG",
    "TGT",
    "TTA",
    "TTC",
    "TTG",
    "TTT",
]


@dataclass(frozen=True)
class ResolvedCondition:
    selected_species: str
    requested_gene: str
    matched_gene_id: str
    matched_species: str
    condition: torch.Tensor


class CfgConditionResolver:
    def __init__(self, species_table: pd.DataFrame, gene_condition_table: pd.DataFrame) -> None:
        if "species" not in species_table.columns:
            raise ValueError("Species reference table must contain a 'species' column.")
        if "species" not in gene_condition_table.columns:
            raise ValueError("Gene-condition table must contain a 'species' column.")
        missing_codon_cols = [col for col in CODON_COLS if col not in gene_condition_table.columns]
        if missing_codon_cols:
            missing_preview = ", ".join(missing_codon_cols[:5])
            raise ValueError(
                "Gene-condition table is missing required codon columns "
                f"(first missing: {missing_preview})."
            )

        self.species_unique = sorted(species_table["species"].dropna().astype(str).unique())
        if not self.species_unique:
            raise ValueError("Species reference table has no usable species values.")
        self._sp2idx = {sp: i for i, sp in enumerate(self.species_unique)}

        self._gene_condition_table = gene_condition_table.copy()
        self._gene_condition_table["species"] = self._gene_condition_table["species"].astype(str)

        self._id_columns = [col for col in ["gene_id", "gene_name", "gene"] if col in self._gene_condition_table.columns]
        if not self._id_columns:
            raise ValueError(
                "Gene-condition table must have at least one gene identifier column: "
                "gene_id, gene_name, or gene."
            )

    def get_species_options(self) -> list[str]:
        return list(self.species_unique)

    def resolve(self, selected_species: str, gene_query: str, expected_condition_dim: int) -> ResolvedCondition:
        species = (selected_species or "").strip()
        gene = (gene_query or "").strip()
        if not species:
            raise ValueError("Please select a species.")
        if species not in self._sp2idx:
            raise ValueError(f"Selected species '{species}' is not present in cfg species reference data.")
        if not gene:
            raise ValueError("Please provide a gene_id or gene name.")

        query_df = self._gene_condition_table.copy()
        match_mask = pd.Series([False] * len(query_df), index=query_df.index)
        gene_lower = gene.lower()
        for col in self._id_columns:
            col_vals = query_df[col].astype(str).str.strip()
            match_mask = match_mask | (col_vals == gene) | (col_vals.str.lower() == gene_lower)

        matched = query_df.loc[match_mask]
        if matched.empty:
            raise ValueError(f"No gene-condition row found for gene '{gene}'.")

        species_matched = matched.loc[matched["species"] == species]
        if species_matched.empty:
            found_species = sorted(matched["species"].astype(str).unique())
            raise ValueError(
                f"Gene '{gene}' exists, but not for selected species '{species}'. "
                f"Found species: {found_species}."
            )

        if len(species_matched) > 1:
            raise ValueError(
                f"Ambiguous gene match for '{gene}' in species '{species}': "
                f"{len(species_matched)} rows found."
            )

        row_data = species_matched.iloc[0]
        codon_info = torch.tensor(
            row_data[CODON_COLS].to_numpy(dtype="float32"),
            dtype=torch.float32,
        ).unsqueeze(0)

        species_idx = self._sp2idx[species]
        species_one_hot = torch.nn.functional.one_hot(
            torch.tensor([species_idx], dtype=torch.long),
            num_classes=len(self.species_unique),
        ).to(torch.float32)

        cond_full = torch.cat([codon_info, species_one_hot], dim=-1)
        if cond_full.shape[1] != expected_condition_dim:
            raise ValueError(
                "Resolved condition dimension mismatch: "
                f"assembled={cond_full.shape[1]}, expected={expected_condition_dim}."
            )

        gene_id_value = str(row_data["gene_id"]) if "gene_id" in species_matched.columns else gene
        return ResolvedCondition(
            selected_species=species,
            requested_gene=gene,
            matched_gene_id=gene_id_value,
            matched_species=str(row_data["species"]),
            condition=cond_full,
        )


@lru_cache(maxsize=1)
def get_cfg_condition_resolver() -> CfgConditionResolver:
    if not config.YEPP_CFG_SPECIES_TABLE.exists():
        raise FileNotFoundError(
            "cfg species reference table not found at "
            f"{config.YEPP_CFG_SPECIES_TABLE}. Set YEPP_CFG_SPECIES_TABLE."
        )
    if not config.YEPP_CFG_GENE_CONDITION_TABLE.exists():
        raise FileNotFoundError(
            "cfg gene-condition table not found at "
            f"{config.YEPP_CFG_GENE_CONDITION_TABLE}. Set YEPP_CFG_GENE_CONDITION_TABLE."
        )

    species_table = pd.read_csv(config.YEPP_CFG_SPECIES_TABLE)
    gene_condition_table = pd.read_csv(config.YEPP_CFG_GENE_CONDITION_TABLE)
    return CfgConditionResolver(species_table=species_table, gene_condition_table=gene_condition_table)
