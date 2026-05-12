"""
INV-5 — Vérifier la mention "10 seeds" / "σ=0.000" dans les .docx
=====================================================================

Objectif:
- Retrouver précisément où apparaît une assertion de reproductibilité
  potentiellement tautologique (10 seeds / std=0.000).

Usage:
  cd "actual_ version"
  python3 investigate_inv5_doc_seed_claim.py

Sortie:
  ../results/resultats_<VERSION_NAME>/inv5_doc_seed_claim_hits.csv
"""

import os
import re
import zipfile
import pandas as pd
from config import CONFIG

SEARCH_TERMS = [
    "10 seeds",
    "10 random seeds",
    "σ=0.000",
    "sigma=0.000",
    "std=0.000",
    "standard deviation of exactly 0.000",
]

VERSION_NAME = CONFIG.get("VERSION_NAME", "trained_models_v9_v6_v4s")
RESULTS_DIR = CONFIG.get("EVAL", {}).get("RESULTS_DIR", f"../results/resultats_{VERSION_NAME}")
OUT_CSV = os.path.join(RESULTS_DIR, "inv5_doc_seed_claim_hits.csv")


def _find_docx_files(root: str):
    docx = []
    for base, _, files in os.walk(root):
        for f in files:
            if f.lower().endswith(".docx") and not f.startswith("~$"):
                docx.append(os.path.join(base, f))
    return sorted(docx)


def _extract_docx_xml_text(docx_path: str):
    hits = []
    with zipfile.ZipFile(docx_path, "r") as zf:
        xml_files = [n for n in zf.namelist() if n.startswith("word/") and n.endswith(".xml")]
        for xf in xml_files:
            raw = zf.read(xf).decode("utf-8", errors="ignore")
            # texte brut simplifié (tags retirés)
            plain = re.sub(r"<[^>]+>", " ", raw)
            plain = re.sub(r"\s+", " ", plain)
            lower = plain.lower()

            for term in SEARCH_TERMS:
                t = term.lower()
                pos = lower.find(t)
                while pos != -1:
                    s = max(0, pos - 120)
                    e = min(len(plain), pos + len(term) + 120)
                    context = plain[s:e].strip()
                    hits.append({
                        "docx": docx_path,
                        "xml_part": xf,
                        "term": term,
                        "context": context,
                    })
                    pos = lower.find(t, pos + 1)
    return hits


def main():
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../.."))
    docx_files = _find_docx_files(repo_root)

    if not docx_files:
        print("Aucun .docx trouvé dans le repo.")
        os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
        pd.DataFrame(columns=["docx", "xml_part", "term", "context"]).to_csv(OUT_CSV, index=False)
        print(f"Saved: {OUT_CSV}")
        return

    print(f"DOCX trouvés: {len(docx_files)}")
    all_hits = []
    for p in docx_files:
        print(f"- Scan: {p}")
        try:
            all_hits.extend(_extract_docx_xml_text(p))
        except Exception as e:
            all_hits.append({
                "docx": p,
                "xml_part": "<ERROR>",
                "term": "<ERROR>",
                "context": str(e),
            })

    out_df = pd.DataFrame(all_hits)
    if out_df.empty:
        out_df = pd.DataFrame(columns=["docx", "xml_part", "term", "context"])

    os.makedirs(os.path.dirname(OUT_CSV), exist_ok=True)
    out_df.to_csv(OUT_CSV, index=False)

    print("\nRésumé INV-5")
    if len(out_df) == 0:
        print("- Aucune occurrence trouvée pour les termes ciblés.")
    else:
        print(f"- Occurrences trouvées: {len(out_df)}")
        print(out_df[["docx", "term"]].value_counts().head(20))

    print(f"\nSaved: {OUT_CSV}")
    print("\nTexte de reformulation recommandé si passage trouvé:")
    print(
        "This result verifies the arithmetic determinism of the SL fusion pipeline at fixed evidence input. "
        "It does not test the reproducibility of the evidence generation stage: Prophet is deterministic by construction, "
        "and RANSAC stochasticity is neutralised by random_state=42. True robustness testing would require varying the "
        "RANSAC seed across multiple training runs (identified as future work §9.3)."
    )


if __name__ == "__main__":
    main()
