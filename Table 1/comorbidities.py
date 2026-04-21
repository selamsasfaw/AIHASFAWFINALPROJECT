import pandas as pd
import os

# -----------------------------
# Load data
# -----------------------------
mimic_path = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1/"

patients = pd.read_csv(os.path.join(mimic_path, "patients.csv"))
diagnoses = pd.read_csv(os.path.join(mimic_path, "diagnoses_icd.csv"))


# diabetes
dm_codes = [
    "25000","25001","25002","25003","25010","25011","25012","25013",
    "E10","E11","E13"
]

# hypertension
htn_codes = [
    # ICD-9
    "4010","4011","4019",
    "40200","40210","40290",
    "40300","40310","40390",
    "40400","40410","40490",
    "40501","40509","40511","40519","40591","40599",

    # ICD-10
    "I10","I11","I12","I13","I15"
]

# heart failure
hf_codes = [
    # ICD-9
    "4280","4281","42820","42821","42822","42823",
    "42830","42831","42832","42833",
    "42840","42841","42842","42843","4289",

    # ICD-10
    "I50","I50.1","I50.2","I50.3","I50.4","I50.9"
]

# hyperlipidemia
hld_codes = [
    "2720","2721","2722","2724",
    "E78"
]

# chronic kidney disease
ckd_codes = [
    "5851","5852","5853","5854","5855","5856",
    "N18"
]

# prior myocardial infarction
mi_codes = [
    "41001","41002","41010","41011","41012","41021","41022",
    "41031","41032","41041","41042","41051","41052",
    "41061","41062","41071","41072","41081","41082","41091","41092",
    "I210","I211","I212","I213","I214"
]


# Create a MACE "label"
mace_ids = diagnoses.loc[
    diagnoses["icd_code"].isin(mi_codes),
    "subject_id"
].unique()

# -----------------------------
# Map comorbidities to patients
# -----------------------------
def map_condition(codes):
    ids = diagnoses.loc[
        diagnoses["icd_code"].isin(codes),
        "subject_id"
    ].unique()
    return patients["subject_id"].isin(ids).astype(int)

patients["DM"] = map_condition(dm_codes)
patients["HTN"] = map_condition(htn_codes)
patients["HF"] = map_condition(hf_codes)
patients["HLD"] = map_condition(hld_codes)
patients["CKD"] = map_condition(ckd_codes)
patients["Prior_MI"] = map_condition(mi_codes)

# mace definition
patients["MACE"] = patients["subject_id"].isin(mace_ids).astype(int)


 
patients["Group"] = patients["MACE"].map({1: "MACE", 0: "Control"})


# build a table for outputs

def summarize(df, group):
    return {
        "Group": group,
        "N": len(df),
        "DM %": df["DM"].mean() * 100,
        "HTN %": df["HTN"].mean() * 100,
        "HF %": df["HF"].mean() * 100,
        "HLD %": df["HLD"].mean() * 100,
        "CKD %": df["CKD"].mean() * 100,
        "Prior MI %": df["Prior_MI"].mean() * 100,
    }

# Build Table 1

overall = summarize(patients, "Overall")
mace = summarize(patients[patients["MACE"] == 1], "MACE")
control = summarize(patients[patients["MACE"] == 0], "Control")

table1 = pd.DataFrame([overall, mace, control])

print(table1)