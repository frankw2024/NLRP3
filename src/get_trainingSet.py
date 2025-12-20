#!/usr/bin/env python3
"""
Fetch all NLRP3 bioactivities from ChEMBL and write a CSV.

- Discovers target_chembl_id by name ("NLRP3", gene "NLRP3", or pref_name).
- Paginates through /activity endpoint.
- Back-fills canonical SMILES via /molecule endpoint.
- Outputs: nlrp3_chembl_activities.csv
"""

import csv
import time
import math
import sys
import requests
from pathlib import Path
from collections import defaultdict

# Get script directory
SCRIPT_DIR = Path(__file__).parent.resolve()

BASE = "https://www.ebi.ac.uk/chembl/api/data"
HEADERS = {"User-Agent": "nlrp3-scraper/1.0 (academic; contact: your_email@example.com)"}
RATE_SLEEP = 0.25  # seconds between calls to be polite

def chembl_get(path, params=None, retries=5):
    url = f"{BASE}/{path}"
    for i in range(retries):
        r = requests.get(url, params=params, headers=HEADERS, timeout=60)
        if r.status_code == 200:
            return r.json()
        if r.status_code in (429, 502, 503, 504):
            time.sleep(1.5 * (i + 1))
            continue
        r.raise_for_status()
    raise RuntimeError(f"Failed GET {url} after {retries} retries")

def find_nlrp3_target_ids(query="NLRP3"):
    """
    Search multiple ways to be robust: target search and gene search.
    Returns a list of (target_chembl_id, pref_name, organism, target_type).
    """
    results = []

    # 1) direct target search by keyword
    data = chembl_get("target", params={"format": "json", "search": query, "limit": 100})
    for t in data.get("targets", []):
        results.append((
            t.get("target_chembl_id"),
            t.get("pref_name"),
            t.get("organism"),
            t.get("target_type"),
        ))

    # 2) xref gene symbol matches (fallback)
    data = chembl_get("target", params={"format": "json", "limit": 100, "target_components__accession__icontains": query})
    for t in data.get("targets", []):
        tup = (t.get("target_chembl_id"), t.get("pref_name"), t.get("organism"), t.get("target_type"))
        if tup not in results:
            results.append(tup)

    # dedupe
    seen = set()
    unique = []
    for row in results:
        if row[0] and row[0] not in seen:
            seen.add(row[0])
            unique.append(row)
    return unique

def fetch_all_activities(target_id, page_limit=1000):
    """
    Paginate through activities for a target.
    Returns a list of dicts from ChEMBL 'activity' records.
    """
    activities = []
    offset = 0
    while True:
        params = {
            "format": "json",
            "target_chembl_id": target_id,
            "limit": page_limit,
            "offset": offset,
        }
        data = chembl_get("activity", params=params)
        page = data.get("activities", [])
        if not page:
            break
        activities.extend(page)
        offset += len(page)
        # Stop if we reached total_count (if provided)
        total = data.get("page_meta", {}).get("total_count")
        if total is not None and offset >= total:
            break
        time.sleep(RATE_SLEEP)
    return activities

def fetch_smiles_for_molecules(molecule_ids):
    """
    Query /molecule for canonical SMILES. Returns dict id->smiles
    """
    id_to_smiles = {}
    ids = list(molecule_ids)
    # Molecule endpoint supports filtering by molecule_chembl_id exact,
    # but not in a single batch; we’ll do small batches for politeness.
    for mid in ids:
        data = chembl_get("molecule", params={"format": "json", "molecule_chembl_id": mid, "limit": 1})
        mols = data.get("molecules", [])
        if mols:
            m = mols[0]
            structs = m.get("molecule_structures") or {}
            smiles = structs.get("canonical_smiles")
            if smiles:
                id_to_smiles[mid] = smiles
        time.sleep(RATE_SLEEP)
    return id_to_smiles

def normalize_activity_row(a):
    """
    Pick useful fields and normalize names.
    """
    return {
        "activity_id": a.get("activity_id"),
        "molecule_chembl_id": a.get("molecule_chembl_id"),
        "assay_chembl_id": a.get("assay_chembl_id"),
        "target_chembl_id": a.get("target_chembl_id"),
        "standard_type": a.get("standard_type"),           # e.g., IC50, Ki, EC50, Inhibition
        "standard_value": a.get("standard_value"),         # numeric as string
        "standard_units": a.get("standard_units"),         # nM, uM, %
        "pchembl_value": a.get("pchembl_value"),           # if available
        "activity_comment": a.get("activity_comment"),
        "relation": a.get("relation"),
        "data_validity_comment": a.get("data_validity_comment"),
        "bao_label": a.get("bao_label"),
        "assay_type": a.get("assay_type"),
        "assay_description": a.get("assay_description"),
        "document_chembl_id": a.get("document_chembl_id"),
        "record_id": a.get("record_id"),
        "src_id": a.get("src_id"),
    }

def main():
    print("Searching ChEMBL for NLRP3 target IDs...")
    candidates = find_nlrp3_target_ids("NLRP3")
    if not candidates:
        print("No NLRP3 targets found in ChEMBL. Exiting.", file=sys.stderr)
        sys.exit(1)

    # Heuristic: prefer HUMAN, SINGLE PROTEIN or PROTEIN COMPLEX targets
    def rank(t):
        tid, name, org, ttype = t
        score = 0
        if org and "Homo sapiens" in org:
            score += 2
        if ttype and ("SINGLE PROTEIN" in ttype or "PROTEIN COMPLEX" in ttype or "PROTEIN FAMILY" in ttype):
            score += 1
        return -score

    candidates_sorted = sorted(candidates, key=rank)
    print("Found possible targets:")
    for tid, name, org, ttype in candidates_sorted:
        print(f"  - {tid} | {name} | {org} | {ttype}")

    target_id = candidates_sorted[0][0]
    print(f"\nUsing target_chembl_id = {target_id}\n")

    print("Fetching activities (this may take a minute)...")
    acts = fetch_all_activities(target_id)
    print(f"Fetched {len(acts)} raw activity rows.")

    # Normalize + dedupe by (molecule_chembl_id, assay_chembl_id, standard_type, standard_value, relation)
    rows = [normalize_activity_row(a) for a in acts]
    seen = set()
    deduped = []
    for r in rows:
        key = (
            r["molecule_chembl_id"],
            r["assay_chembl_id"],
            r["standard_type"],
            r["standard_value"],
            r["relation"],
        )
        if key not in seen:
            seen.add(key)
            deduped.append(r)
    print(f"After deduplication: {len(deduped)} rows.")

    # Fetch canonical SMILES for unique molecules
    mol_ids = {r["molecule_chembl_id"] for r in deduped if r["molecule_chembl_id"]}
    print(f"Fetching canonical SMILES for {len(mol_ids)} unique molecules...")
    id2smiles = fetch_smiles_for_molecules(mol_ids)

    # Attach SMILES
    for r in deduped:
        r["canonical_smiles"] = id2smiles.get(r["molecule_chembl_id"])

    out_csv = SCRIPT_DIR / "nlrp3_chembl_activities.csv"
    fieldnames = [
        "molecule_chembl_id", "canonical_smiles",
        "activity_id", "assay_chembl_id", "target_chembl_id",
        "standard_type", "standard_value", "standard_units",
        "pchembl_value", "relation", "bao_label",
        "assay_type", "assay_description",
        "activity_comment", "data_validity_comment",
        "document_chembl_id", "record_id", "src_id",
    ]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in deduped:
            w.writerow(r)

    print(f"\nWrote {len(deduped)} rows to {out_csv}")
    print("\nTip:")
    print("- Filter to quantitative types (IC50, Ki, EC50) and valid standard_units (nM, uM).")
    print("- Prefer rows with pchembl_value present for comparability.")
    print("- Use Bemis–Murcko scaffolds for train/val/test splits to avoid leakage.")

if __name__ == "__main__":
    main()
