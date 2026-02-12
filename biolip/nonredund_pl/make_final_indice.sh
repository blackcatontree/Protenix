#!/usr/bin/env bash
set -euo pipefail

# Paths (fixed)
IDX_IN="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_ligand_prot_clean2.csv"
BIO_DIR="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/bioassembly"
BAD_LIST="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/select.txt"
IDX_OUT="/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_final.csv"

echo "=== Build indices_final.csv by atom_array(hetero) ground truth ==="
echo "[IN ] IDX_IN  : $IDX_IN"
echo "[IN ] BIO_DIR : $BIO_DIR"
echo "[IN ] BADLIST : $BAD_LIST"
echo "[OUT] IDX_OUT : $IDX_OUT"

python - <<'PY'
import gzip, pickle
import pandas as pd
import numpy as np
from pathlib import Path

IDX_IN = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_ligand_prot_clean2.csv")
BIO_DIR = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/bioassembly")
BAD_PATH = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/select.txt")
IDX_OUT = Path("/home/dataset-local/tmp/zsl/Protenix/biolip/nonredund_pl/all_data/prepared/indices_final.csv")

# blacklist
bad = set()
for line in BAD_PATH.read_text().splitlines():
    line = line.strip()
    if (not line) or line.startswith("#"):
        continue
    bad.add(line.split()[0].upper())
bad |= {"HOH","WAT","DOD","H2O",".","?"}

df = pd.read_csv(IDX_IN)
# enforce interface ligand_prot
df = df[(df["mol_type_group"]=="ligand_prot") & (df["type"]=="interface")].copy()

df["pdb_u"] = df["pdb_id"].astype(str).str.lower()
df["lig_u"] = df["cluster_2_id"].astype(str).str.upper()

pdbs = df["pdb_u"].unique().tolist()
pdb_to_ligs = {}
missing = 0
bad_atomarray = 0

for pdb in pdbs:
    pkl = BIO_DIR / f"{pdb}.pkl.gz"
    if not pkl.exists():
        pdb_to_ligs[pdb] = set()
        missing += 1
        continue

    try:
        with gzip.open(pkl, "rb") as f:
            obj = pickle.load(f)
        aa = obj.get("atom_array", None)
        if aa is None or not (hasattr(aa,"res_name") and hasattr(aa,"hetero")):
            pdb_to_ligs[pdb] = set()
            bad_atomarray += 1
            continue

        res = np.asarray(aa.res_name).astype(str)
        het = np.asarray(aa.hetero).astype(bool)

        ligs = set(res[het])
        ligs = set([x.upper() for x in ligs if x and (x.upper() not in bad)])
        pdb_to_ligs[pdb] = ligs
    except Exception:
        pdb_to_ligs[pdb] = set()
        bad_atomarray += 1

keep = [lig in pdb_to_ligs.get(pdb, set()) for pdb, lig in zip(df["pdb_u"], df["lig_u"])]
df2 = df[pd.Series(keep, index=df.index)].copy()
df2.drop(columns=["pdb_u","lig_u"], inplace=True)
df2.to_csv(IDX_OUT, index=False)

print("input_rows:", len(df))
print("kept_rows :", len(df2))
print("removed_rows:", len(df)-len(df2))
print("unique_pdb:", len(pdbs))
print("missing_pkl:", missing)
print("bad_atomarray_or_load_fail:", bad_atomarray)
print("wrote:", IDX_OUT)
PY

echo ""
echo "=== DONE ==="
echo "[OUT] $IDX_OUT"
echo ""
echo "Next step:"
echo "1) Set configs/configs_data.py -> qbiolip_nonredund.base_info.indices_fpath to:"
echo "   $IDX_OUT"
echo "2) Re-run training."
