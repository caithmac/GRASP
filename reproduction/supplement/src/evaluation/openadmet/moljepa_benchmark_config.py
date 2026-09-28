"""Locked metadata for the Mol-JEPA 23-endpoint regression benchmark.

The raw tables are public, but the authors' ``*_clustersplits.sdf`` files are
not part of the Mol-JEPA repository.  This module keeps raw-source provenance,
paper row counts, paper split sizes, and Table 3 reference values in one place
so reconstructed and exact-SDF runs cannot be confused.
"""
from __future__ import annotations

from dataclasses import dataclass


PAPER_URL = "https://arxiv.org/abs/2608.22642v2"
PAPER_PROTOCOL = {
    "fingerprint": "ECFP4",
    "radius": 2,
    "n_bits": 1024,
    "tanimoto_threshold": 0.65,
    "splits": ("split1", "split2", "split3"),
}


RAW_FILES = {
    "expansion": {
        "filename": "expansion_data_train.csv",
        "url": (
            "https://huggingface.co/datasets/openadmet/"
            "openadmet-expansionrx-challenge-data/resolve/"
            "6b898ccc43d10d25b230fb09e22a6e30c30022b5/"
            "expansion_data_train.csv"
        ),
        "sha256": "c5214ad8c8a4d7d4d09082fcce24fe16c97a980a1e0873b91b8c8e474b79f6e4",
        "revision": "6b898ccc43d10d25b230fb09e22a6e30c30022b5",
    },
    "asap_admet": {
        "filename": "ASAP_ADMET.csv",
        "url": (
            "https://huggingface.co/datasets/openadmet/"
            "ASAP_Polaris_OpenADMET_challenge/resolve/"
            "060aee45a3cf2ab030b325e16c301a06d2d19d71/ADMET.csv"
        ),
        "sha256": "3388480e9a312c17c9b621352f9414b0456ac19fc44604570e82a112e2200603",
        "revision": "060aee45a3cf2ab030b325e16c301a06d2d19d71",
    },
    "asap_potency": {
        "filename": "ASAP_Potency.csv",
        "url": (
            "https://huggingface.co/datasets/openadmet/"
            "ASAP_Polaris_OpenADMET_challenge/resolve/"
            "060aee45a3cf2ab030b325e16c301a06d2d19d71/Potency.csv"
        ),
        "sha256": "2c7e877684e01ba079552aff302e74cda89023e80c966c995748336d3cf1e5ed",
        "revision": "060aee45a3cf2ab030b325e16c301a06d2d19d71",
    },
    "pxr": {
        "filename": "PXR_train.csv",
        "url": (
            "https://huggingface.co/datasets/openadmet/"
            "pxr-challenge-train-test/resolve/"
            "c320ab14df0ed66a485b01839e9db8874f81509b/"
            "pxr-challenge_TRAIN.csv"
        ),
        "sha256": "efa9096110c9123386141cb505507d10f3de845c764d07e9c7375f644c82757c",
        "revision": "c320ab14df0ed66a485b01839e9db8874f81509b",
    },
    "biogen": {
        "filename": "ADME_public_set_3521.csv",
        "url": (
            "https://raw.githubusercontent.com/molecularinformatics/"
            "Computational-ADME/b00df003de117ce9e5b381afd886095c5f2af2d5/"
            "ADME_public_set_3521.csv"
        ),
        "sha256": "2cfabc2667740c224487876c33b23124159ef43294e0f9e4d926cb6276c95a3b",
        "revision": "b00df003de117ce9e5b381afd886095c5f2af2d5",
    },
}


REFERENCE_MODELS = (
    "Mol-JEPA Best",
    "Mol-JEPA Transformer",
    "CLAMP Nonlinear",
    "CheMeleon Finetuned",
    "Chemprop Finetuned",
    "TabICLv2 AlvaDesc",
    "RF ECFP4",
    "LGBM AlvaDesc",
)


@dataclass(frozen=True)
class Endpoint:
    slug: str
    family: str
    display_name: str
    source: str
    smiles_column: str
    target_column: str
    transform: str
    paper_total: int
    paper_test_sizes: tuple[int, int, int]
    exact_sdf_relpath: str
    reference: tuple[tuple[float, float], ...]
    source_filter_column: str | None = None
    source_filter_value: str | None = None


def _r(*values: tuple[float, float]) -> tuple[tuple[float, float], ...]:
    if len(values) != len(REFERENCE_MODELS):
        raise ValueError("every endpoint needs all eight Table 3 references")
    return values


ENDPOINTS = (
    Endpoint(
        "expansion_caco2_pappa", "ExpansionRx", "Caco-2 Permeability", "expansion",
        "SMILES", "Caco-2 Permeability Papp A>B", "positive_log10", 2156,
        (238, 214, 202),
        "openadmet/Caco2_Perm_PappA/openadmet_Caco-2_Permeability_Papp_A_clustersplits.sdf",
        _r((.34,.02),(.35,.02),(.40,.05),(.37,.05),(.39,.04),(.42,.02),(.43,.02),(.42,.02)),
    ),
    Endpoint(
        "expansion_caco2_efflux", "ExpansionRx", "Caco-2 Efflux", "expansion",
        "SMILES", "Caco-2 Permeability Efflux", "positive_log10", 2161,
        (339, 260, 208),
        "openadmet/Caco2_Perm_Efflux/openadmet_Caco-2_Permeability_Efflux_clustersplits.sdf",
        _r((.25,.02),(.26,.01),(.31,.03),(.27,.05),(.24,.03),(.24,.02),(.28,.04),(.24,.02)),
    ),
    Endpoint(
        "expansion_logd", "ExpansionRx", "LogD", "expansion", "SMILES", "LogD",
        "identity", 5039, (534, 570, 551),
        "openadmet/LogD/openadmet_LogD_clustersplits.sdf",
        _r((.33,.02),(.46,.04),(.50,.05),(.35,.02),(.74,.06),(.36,.02),(.73,.07),(.43,.04)),
    ),
    Endpoint(
        "expansion_ksol", "ExpansionRx", "KSOL", "expansion", "SMILES", "KSOL",
        "positive_log10", 5128, (608, 496, 520),
        "openadmet/KSOL/openadmet_KSOL_clustersplits.sdf",
        _r((.45,.03),(.49,.04),(.53,.04),(.45,.06),(.60,.05),(.46,.06),(.62,.04),(.49,.03)),
    ),
    Endpoint(
        "expansion_hlm", "ExpansionRx", "HLM CLint", "expansion", "SMILES", "HLM CLint",
        "positive_log10", 3595, (443, 409, 426),
        "openadmet/HLM_CLint/openadmet_HLM_CLint_clustersplits.sdf",
        _r((.39,.02),(.43,.03),(.50,.02),(.40,.06),(.46,.01),(.39,.05),(.47,.02),(.39,.03)),
    ),
    Endpoint(
        "expansion_mlm", "ExpansionRx", "MLM CLint", "expansion", "SMILES", "MLM CLint",
        "positive_log10", 4375, (617, 528, 655),
        "openadmet/MLM_CLint/openadmet_MLM_CLint_clustersplits.sdf",
        _r((.40,.05),(.38,.07),(.36,.04),(.39,.06),(.42,.04),(.41,.09),(.44,.08),(.38,.07)),
    ),
    Endpoint(
        "expansion_mbpb", "ExpansionRx", "MBPB", "expansion", "SMILES", "MBPB",
        "positive_log10", 973, (120, 113, 176),
        "openadmet/MBPB/openadmet_MBPB_clustersplits.sdf",
        _r((.30,.02),(.30,.03),(.31,.01),(.29,.06),(.45,.10),(.26,.03),(.43,.01),(.34,.04)),
    ),
    Endpoint(
        "expansion_mgmb", "ExpansionRx", "MGMB", "expansion", "SMILES", "MGMB",
        "positive_log10", 221, (15, 22, 44),
        "openadmet/MGMB/openadmet_MGMB_clustersplits.sdf",
        _r((.30,.05),(.28,.06),(.35,.08),(.31,.13),(.41,.13),(.30,.09),(.39,.12),(.33,.10)),
    ),
    Endpoint(
        "expansion_mppb", "ExpansionRx", "MPPB", "expansion", "SMILES", "MPPB",
        "positive_log10", 1292, (133, 147, 146),
        "openadmet/MPPB/openadmet_MPPB_clustersplits.sdf",
        _r((.26,.04),(.28,.05),(.31,.03),(.31,.02),(.41,.01),(.26,.05),(.41,.06),(.29,.05)),
    ),
    Endpoint(
        "asap_mers", "ASAP", "MERS-CoV-2 Potency", "asap_potency", "CXSMILES",
        "pIC50 (MERS-CoV Mpro)", "identity", 421, (38, 55, 55),
        "asap_potency/MERS/asap_potency_MERS_clustersplits.sdf",
        _r((.59,.10),(.70,.08),(.96,.04),(.86,.09),(.52,.08),(.54,.09),(.58,.04),(.59,.13)),
        "Set", "Train",
    ),
    Endpoint(
        "asap_sars", "ASAP", "SARS-CoV-2 Potency", "asap_potency", "CXSMILES",
        "pIC50 (SARS-CoV-2 Mpro)", "identity", 356, (29, 49, 21),
        "asap_potency/SARS/asap_potency_SARS_clustersplits.sdf",
        _r((.62,.06),(.62,.06),(1.02,.12),(.71,.13),(.73,.18),(.64,.17),(.77,.25),(.68,.16)),
        "Set", "Train",
    ),
    Endpoint(
        "asap_logd", "ASAP", "LogD", "asap_admet", "CXSMILES", "LogD", "identity",
        203, (32, 40, 34), "asap_admet/LogD/LogD_clustersplits.sdf",
        _r((.68,.09),(.72,.11),(1.09,.14),(.84,.19),(.64,.14),(.89,.07),(.99,.05),(.74,.14)),
        "Set", "Train",
    ),
    Endpoint(
        "asap_ksol", "ASAP", "KSOL", "asap_admet", "CXSMILES", "KSOL",
        "positive_log10", 214, (31, 28, 29), "asap_admet/KSOL/log_KSOL_clustersplits.sdf",
        _r((.42,.19),(.53,.08),(.91,.19),(.60,.02),(.67,.19),(.65,.12),(.58,.06),(.70,.11)),
        "Set", "Train",
    ),
    Endpoint(
        "asap_hlm", "ASAP", "HLM", "asap_admet", "CXSMILES", "HLM",
        "positive_log10", 166, (18, 13, 13), "asap_admet/HLM/log_HLM_clustersplits.sdf",
        _r((.39,.11),(.39,.11),(.58,.03),(.50,.24),(.36,.09),(.42,.15),(.43,.07),(.43,.10)),
        "Set", "Train",
    ),
    Endpoint(
        "asap_mlm", "ASAP", "MLM", "asap_admet", "CXSMILES", "MLM",
        "positive_log10", 189, (39, 32, 44), "asap_admet/MLM/log_MLM_clustersplits.sdf",
        _r((.53,.01),(.53,.01),(1.02,.37),(.67,.13),(.64,.08),(.59,.15),(.50,.06),(.51,.09)),
        "Set", "Train",
    ),
    Endpoint(
        "asap_mdr1", "ASAP", "MDR1 Efflux", "asap_admet", "CXSMILES", "MDR1-MDCKII",
        "positive_log10", 242, (32, 37, 29),
        "asap_admet/MDR1/log_MDR1-MDCKII_clustersplits.sdf",
        _r((.42,.13),(.44,.14),(.74,.14),(.51,.30),(.45,.22),(.48,.26),(.56,.36),(.58,.41)),
        "Set", "Train",
    ),
    Endpoint(
        "pxr", "PXR", "PXR Activity", "pxr", "SMILES", "pEC50", "identity", 4288,
        (245, 233, 236), "PXR/PXR_clustersplits.sdf",
        _r((.55,.04),(.55,.06),(.75,.06),(.59,.13),(.79,.11),(.87,.20),(.81,.11),(.87,.17)),
    ),
    Endpoint(
        "biogen_solubility", "Biogen ADME", "Solubility", "biogen", "SMILES",
        "LOG SOLUBILITY PH 6.8 (ug/mL)", "identity", 2173, (98, 113, 125),
        "biogen/solubility/biogen_LOG_SOLUBILITY_clustersplits.sdf",
        _r((.30,.02),(.30,.02),(.38,.02),(.35,.03),(.42,.04),(.33,.01),(.43,.01),(.33,.01)),
    ),
    Endpoint(
        "biogen_hlm", "Biogen ADME", "HLM CLint", "biogen", "SMILES",
        "LOG HLM_CLint (mL/min/kg)", "identity", 3087, (104, 75, 75),
        "biogen/HLM_CLint/biogen_HLM_CLint_clustersplits.sdf",
        _r((.33,.02),(.34,.02),(.45,.06),(.35,.01),(.51,.09),(.37,.03),(.55,.08),(.36,.02)),
    ),
    Endpoint(
        "biogen_rlm", "Biogen ADME", "RLM CLint", "biogen", "SMILES",
        "LOG RLM_CLint (mL/min/kg)", "identity", 3054, (107, 124, 104),
        "biogen/RLM_CLint/biogen_RLM_CLint_clustersplits.sdf",
        _r((.37,.04),(.38,.04),(.46,.04),(.39,.01),(.57,.07),(.39,.03),(.56,.04),(.39,.04)),
    ),
    Endpoint(
        "biogen_hppb", "Biogen ADME", "HPPB", "biogen", "SMILES",
        "LOG PLASMA PROTEIN BINDING (HUMAN) (% unbound)", "identity", 194, (25, 20, 21),
        "biogen/PPBhuman/biogen_PPBhuman_clustersplits.sdf",
        _r((.33,.06),(.34,.07),(.45,.05),(.48,.08),(.56,.07),(.35,.04),(.67,.11),(.40,.07)),
    ),
    Endpoint(
        "biogen_rppb", "Biogen ADME", "RPPB", "biogen", "SMILES",
        "LOG PLASMA PROTEIN BINDING (RAT) (% unbound)", "identity", 168, (20, 20, 20),
        "biogen/PPBrat/biogen_PPBrat_clustersplits.sdf",
        _r((.43,.07),(.45,.08),(.48,.18),(.55,.13),(.57,.18),(.49,.12),(.68,.15),(.56,.12)),
    ),
    Endpoint(
        "biogen_mdr1", "Biogen ADME", "MDR1 Efflux", "biogen", "SMILES",
        "LOG MDR1-MDCK ER (B-A/A-B)", "identity", 2642, (77, 75, 76),
        "biogen/MDR1/biogen_MDR1_MDCK_clustersplits.sdf",
        _r((.27,.03),(.29,.01),(.38,.05),(.30,.06),(.39,.02),(.31,.04),(.55,.08),(.32,.04)),
    ),
)


ENDPOINT_BY_SLUG = {endpoint.slug: endpoint for endpoint in ENDPOINTS}
if len(ENDPOINT_BY_SLUG) != 23:
    raise RuntimeError("Mol-JEPA benchmark manifest must contain exactly 23 unique endpoints")

