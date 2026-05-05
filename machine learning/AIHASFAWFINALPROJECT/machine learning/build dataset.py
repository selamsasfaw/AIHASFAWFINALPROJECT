import pandas as pd
import os

# load data
mimic_path = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1/"

patients = pd.read_csv(os.path.join(mimic_path, "patients.csv"))
diagnoses = pd.read_csv(os.path.join(mimic_path, "diagnoses_icd.csv"))

#icd code cleanup
diagnoses["icd_code"] = (
    diagnoses["icd_code"]
    .astype(str)
    .str.upper()
    .str.strip()
    .str.replace(".", "", regex=False)
)

#define the cohort
cohort = patients[["subject_id", "gender", "anchor_age"]].copy()

#encode sex as binary
cohort["male"] = (cohort["gender"] == "M").astype(int)
cohort.drop(columns=["gender"], inplace=True)

#fill missing ages with median (NOT MEAN)
cohort["anchor_age"] = cohort["anchor_age"].fillna(cohort["anchor_age"].median())

# icd code groups for comorbidities and outcomes
dm_codes = ["250", "E10", "E11", "E13"]
htn_codes = ["401", "402", "I10", "I11", "I12", "I13", "I15"]
hf_codes = ["428", "I50"]
hld_codes = ["272", "E78"]
ckd_codes = ["585", "N18"]
mi_codes = ["410", "I21"]  # simplified MI codes (acute MI)

# feautre creation to shown that the diagnosis is present in the a given patient history
def make_feature(code_prefixes):
    ids = diagnoses.loc[
        diagnoses["icd_code"].str.startswith(tuple(code_prefixes)),
        "subject_id"
    ].unique()
    return cohort["subject_id"].isin(ids).astype(int)

# comorbidities
cohort["DM"] = make_feature(dm_codes)
cohort["HTN"] = make_feature(htn_codes)
cohort["HF"] = make_feature(hf_codes)
cohort["HLD"] = make_feature(hld_codes)
cohort["CKD"] = make_feature(ckd_codes)

# MACE outcome (MI+ STROKE+ DEATH!)
mace_prefixes = mi_codes + hf_codes + ["I63", "I64", "434"]

mace_ids = diagnoses.loc[
    diagnoses["icd_code"].str.startswith(tuple(mace_prefixes)),
    "subject_id"
].unique()

cohort["MACE"] = cohort["subject_id"].isin(mace_ids).astype(int)

# final cleanup
cohort = cohort.drop_duplicates(subset=["subject_id"])

binary_cols = ["male", "DM", "HTN", "HF", "HLD", "CKD", "MACE"]
cohort[binary_cols] = cohort[binary_cols].astype(int)

# final ML dataset
ml_df = cohort[
    [
        "anchor_age",
        "male",
        "DM",
        "HTN",
        "HF",
        "HLD",
        "CKD",
        "MACE"
    ]
]

# save dataset (TRIAL 2)
ml_df.to_csv("ml_dataset.csv", index=False)