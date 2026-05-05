import pandas as pd
import os

# load data
mimic_path = "/oscar/data/shared/ursa/mimic-iv/hosp/3.1/"

patients = pd.read_csv(os.path.join(mimic_path, "patients.csv"))

diagnoses = pd.read_csv(
    os.path.join(mimic_path, "diagnoses_icd.csv"),
    usecols=['subject_id', 'icd_code']
)

#define MACE codes
mace_icd_codes = [
    # Myocardial Infarction (ICD-9)
    '41001','41002','41010','41011','41012','41021','41022','41031','41032',
    '41040','41041','41042','41051','41052','41061','41062','41070','41071', 
    '41081','41082','41090','41091','41092',

    # Myocardial Infarction (ICD-10)
    'I210','I211','I212','I213','I214',

    # Stroke (ICD-9)
    '43400','43401','43410','43411','43490','43491',

    # Stroke (ICD-10)
    'I63','I64'
]

mace_ids = diagnoses[diagnoses['icd_code'].isin(mace_icd_codes)]['subject_id'].unique()
patients['MACE'] = patients['subject_id'].isin(mace_ids).astype(int)

#age range I am interested in
patients['age_lt_50'] = patients['anchor_age'] < 50


def count_pct(series, mask=None):
    if mask is not None:
        series = series[mask]

    n = series.sum()
    denom = len(series)

    pct = (n / denom * 100) if denom > 0 else 0
    return n, pct

# calculations
mace_mask = patients['MACE'] == 1
control_mask = patients['MACE'] == 0

total_n, total_pct = count_pct(patients['age_lt_50'])
mace_n, mace_pct = count_pct(patients['age_lt_50'], mace_mask)
control_n, control_pct = count_pct(patients['age_lt_50'], control_mask)

# form outputs
print("Age <50:")
print(f"  Total:   {total_n} ({total_pct:.1f}%)")
print(f"  MACE:    {mace_n} ({mace_pct:.1f}%)")
print(f"  Control: {control_n} ({control_pct:.1f}%)")


# Age 50–59 variable
age_50_59 = (patients['anchor_age'] >= 50) & (patients['anchor_age'] <= 59)

# total
total_n = age_50_59.sum()
total_pct = total_n / len(patients) * 100

# MACE
mace_n = age_50_59[mace_mask].sum()
mace_pct = mace_n / mace_mask.sum() * 100 if mace_mask.sum() > 0 else 0

# control
control_n = age_50_59[control_mask].sum()
control_pct = control_n / control_mask.sum() * 100 if control_mask.sum() > 0 else 0

# output
print("Age 50–59:")
print(f"  Total:   {total_n} ({total_pct:.1f}%)")
print(f"  MACE:    {mace_n} ({mace_pct:.1f}%)")
print(f"  Control: {control_n} ({control_pct:.1f}%)")


# Age 60–69 variable
age_60_69 = (patients['anchor_age'] >= 60) & (patients['anchor_age'] <= 69)

# total
total_n = age_60_69.sum()
total_pct = total_n / len(patients) * 100

# MACE
mace_n = age_60_69[mace_mask].sum()
mace_pct = mace_n / mace_mask.sum() * 100 if mace_mask.sum() > 0 else 0

# control
control_n = age_60_69[control_mask].sum()
control_pct = control_n / control_mask.sum() * 100 if control_mask.sum() > 0 else 0

# output
print("Age 60–69:")
print(f"  Total:   {total_n} ({total_pct:.1f}%)")
print(f"  MACE:    {mace_n} ({mace_pct:.1f}%)")
print(f"  Control: {control_n} ({control_pct:.1f}%)")

# Age ≥70 variable
age_70_plus = patients['anchor_age'] >= 70

# total
total_n = age_70_plus.sum()
total_pct = total_n / len(patients) * 100

# MACE
mace_n = age_70_plus[mace_mask].sum()
mace_pct = mace_n / mace_mask.sum() * 100 if mace_mask.sum() > 0 else 0


# CONTROL
control_n = age_70_plus[control_mask].sum()
control_pct = control_n / control_mask.sum() * 100 if control_mask.sum() > 0 else 0


# OUTPUT
print("Age ≥70:")
print(f"  Total:   {total_n} ({total_pct:.1f}%)")
print(f"  MACE:    {mace_n} ({mace_pct:.1f}%)")
print(f"  Control: {control_n} ({control_pct:.1f}%)")
