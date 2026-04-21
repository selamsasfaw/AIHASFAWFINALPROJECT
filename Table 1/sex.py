import pandas as pd
import os

# -----------------------------
# Load MIMIC data
# -----------------------------
mimic_path = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1/"

patients = pd.read_csv(os.path.join(mimic_path, "patients.csv"))
diagnoses = pd.read_csv(os.path.join(mimic_path, "diagnoses_icd.csv"))

# -----------------------------
# Define MACE ICD codes
# -----------------------------
mi_icd9 = [
    "41001","41002","41010","41011","41012","41021","41022",
    "41031","41032","41041","41042","41051","41052",
    "41061","41062","41071","41072","41081","41082","41091","41092"
]

mi_icd10 = ["I210","I211","I212","I213","I214"]

stroke_icd9 = ["43400","43401","43410","43411","43490","43491"]
stroke_icd10 = ["I63","I64"]

mace_icd_codes = mi_icd9 + mi_icd10 + stroke_icd9 + stroke_icd10

# -----------------------------
# Create MACE patient flag
# -----------------------------
mace_ids = diagnoses.loc[
    diagnoses["icd_code"].isin(mace_icd_codes),
    "subject_id"
].unique()

patients["MACE"] = patients["subject_id"].isin(mace_ids).astype(int)

# -----------------------------
# Create sex variables
# -----------------------------
patients["male"] = (patients["gender"] == "M")

# -----------------------------
# Helper function for Table 1
# -----------------------------
def summarize(group_df, label):
    total = len(group_df)
    male = group_df["male"].sum()
    female = total - male

    mace_total = group_df["MACE"].sum()
    mace_rate = mace_total / total * 100

    return {
        "Group": label,
        "N": total,
        "Male n (%)": f"{male} ({male/total*100:.1f}%)",
        "Female n (%)": f"{female} ({female/total*100:.1f}%)",
        "MACE n (%)": f"{mace_total} ({mace_rate:.1f}%)"
    }

# -----------------------------
# Compute groups
# -----------------------------
overall = summarize(patients, "Overall")
mace_group = summarize(patients[patients["MACE"] == 1], "MACE")
control_group = summarize(patients[patients["MACE"] == 0], "Control")

# -----------------------------
# Print Table 1-style output
# -----------------------------
table1 = pd.DataFrame([overall, mace_group, control_group])

print(table1)