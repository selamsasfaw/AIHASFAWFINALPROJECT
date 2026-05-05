import pandas as pd
import os

# -----------------------------
# Load data
# -----------------------------
mimic_path = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1/"

patients = pd.read_csv(os.path.join(mimic_path, "patients.csv"))
diagnoses = pd.read_csv(os.path.join(mimic_path, "diagnoses_icd.csv"))

# clean ICD codes
diagnoses["icd_code"] = (
    diagnoses["icd_code"]
    .astype(str)
    .str.upper()
    .str.replace(".", "", regex=False)
)

# -----------------------------
# DEFINE CONDITIONS (prefix-based)
# -----------------------------

def has_prefix(prefixes):
    ids = diagnoses[
        diagnoses["icd_code"].str.startswith(prefixes)
    ]["subject_id"].unique()
    return patients["subject_id"].isin(ids).astype(int)

# comorbidities
patients["DM"]  = has_prefix(("250", "E10", "E11", "E13"))
patients["HTN"] = has_prefix(("401", "402", "403", "404", "405", "I10", "I11", "I12", "I13", "I15"))
patients["HF"]  = has_prefix(("428", "I50"))
patients["HLD"] = has_prefix(("272", "E78"))
patients["CKD"] = has_prefix(("585", "N18"))

# -----------------------------
# MACE (NO Prior_MI variable)
# -----------------------------
mace_prefixes = (
    "410",   # MI ICD-9
    "I21",   # MI ICD-10
    "434",   # stroke ICD-9
    "I63",   # stroke ICD-10
    "I64"
)

mace_ids = diagnoses[
    diagnoses["icd_code"].str.startswith(mace_prefixes)
]["subject_id"].unique()

patients["MACE"] = patients["subject_id"].isin(mace_ids).astype(int)

# -----------------------------
# GROUP LABEL
# -----------------------------
patients["Group"] = patients["MACE"].map({1: "MACE", 0: "Control"})

# -----------------------------
# SUMMARY TABLE
# -----------------------------
def summarize(df, group):
    return {
        "Group": group,
        "N": len(df),
        "DM %": df["DM"].mean() * 100,
        "HTN %": df["HTN"].mean() * 100,
        "HF %": df["HF"].mean() * 100,
        "HLD %": df["HLD"].mean() * 100,
        "CKD %": df["CKD"].mean() * 100
    }

overall = summarize(patients, "Overall")
mace = summarize(patients[patients["MACE"] == 1], "MACE")
control = summarize(patients[patients["MACE"] == 0], "Control")

table1 = pd.DataFrame([overall, mace, control])

print(table1)