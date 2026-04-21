import pandas as pd
import os


mimic_path = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1/"

patients = pd.read_csv(os.path.join(mimic_path, "patients.csv"))
diagnoses = pd.read_csv(os.path.join(mimic_path, "diagnoses_icd.csv"))


diagnoses["icd_code"] = (
    diagnoses["icd_code"]
    .astype(str)
    .str.upper()
    .str.strip()
    .str.replace(".", "", regex=False)
)

#create patient table based on input suited for XGBoost
cohort = patients[["subject_id", "gender", "anchor_age"]].copy()

# add sex variable
cohort["male"] = (cohort["gender"] == "M").astype(int)
cohort.drop(columns=["gender"], inplace=True)

# If any patient’s age is missing, replace it with the median age in the dataset (need to review)
cohort["anchor_age"] = cohort["anchor_age"].fillna(cohort["anchor_age"].median())

#redefine icd codes

dm_codes = [
    "25000","25001","25002","25003","25010","25011","25012","25013",
    "E10","E11","E13"
]

htn_codes = [
    "4010","4011","4019","40200","40210","40290",
    "I10","I11","I12","I13","I15"
]

hf_codes = [
    "4280","4281","42820","42821","42822","42823",
    "42830","42831","42832","42833",
    "42840","42841","42842","42843","4289",
    "I50","I501","I502","I503","I504","I509"
]

hld_codes = [
    "2720","2721","2722","2724",
    "E78"
]

ckd_codes = [
    "5851","5852","5853","5854","5855","5856",
    "N18"
]

mi_codes = [
    "41001","41002","41010","41011","41012","41021","41022",
    "41031","41032","41041","41042","41051","41052",
    "41061","41062","41071","41072","41081","41082","41091","41092",
    "I210","I211","I212","I213","I214"
]

#creates the binary flag that tells if you if each patient has a specific condition (REMOVE DUPLICATES)
def make_feature(code_list):
    ids = diagnoses.loc[
        diagnoses["icd_code"].isin(code_list),
        "subject_id"
    ].unique()
    return cohort["subject_id"].isin(ids).astype(int)
#note: list of all patients who have at least one ICD code in code_list


#comorbidities definied
cohort["DM"] = make_feature(dm_codes)
cohort["HTN"] = make_feature(htn_codes)
cohort["HF"] = make_feature(hf_codes)
cohort["HLD"] = make_feature(hld_codes)
cohort["CKD"] = make_feature(ckd_codes)
cohort["Prior_MI"] = make_feature(mi_codes)

#define MACE (used a broader definiton, don't have too many patients)
mace_codes = mi_codes + hf_codes + ["I63", "I64", "43491", "43411"]

mace_ids = diagnoses.loc[
    diagnoses["icd_code"].isin(mace_codes),
    "subject_id"
].unique()

cohort["MACE"] = cohort["subject_id"].isin(mace_ids).astype(int)

#keep only the first record for each patient and remove any extra rows that have the same subject ID
cohort = cohort.drop_duplicates(subset=["subject_id"])

binary_cols = ["male", "DM", "HTN", "HF", "HLD", "CKD", "Prior_MI", "MACE"]
cohort[binary_cols] = cohort[binary_cols].astype(int)

#output final ML dataset
ml_df = cohort[
    [
        "anchor_age",
        "male",
        "DM",
        "HTN",
        "HF",
        "HLD",
        "CKD",
        "Prior_MI",
        "MACE"
    ]
]

#save dataset as CSV file
ml_df.to_csv("ml_dataset.csv", index=False)

