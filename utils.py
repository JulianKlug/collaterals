import pandas as pd
import os
import numpy as np

def remove_french_accents_and_cedillas_from_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = df.select_dtypes(include=[np.object]).columns
    df[cols] = df[cols].apply(
        lambda x: x.str.normalize('NFKD').str.encode('ascii', errors='ignore').str.decode('utf-8'))

    return df

def create_ehr_case_identification_column(df):
    # Identify each case with case id (patient id + eds last 4 digits)
    case_identification_column = df['patient_id'].astype(str) \
                                 + '_' + df['eds_end_4digit'].str.zfill(4).astype(str)
    return case_identification_column


def create_registry_case_identification_column(df):
    # Identify each case with case id (patient id + eds last 4 digits)
    df = df.copy()
    if 'patient_id' not in df.columns:
        df['patient_id'] = df['Case ID'].apply(lambda x: x[8:-4]).astype(str)
    if 'EDS_last_4_digits' not in df.columns:
        df['EDS_last_4_digits'] = df['Case ID'].apply(lambda x: x[-4:]).astype(str)
    case_identification_column = df['patient_id'].astype(str) \
                                 + '_' + df['EDS_last_4_digits'].str.zfill(4).astype(str)
    return case_identification_column


def load_data_from_main_dir(data_path: str, file_start: str) -> pd.DataFrame:
    files = [pd.read_csv(os.path.join(data_path, f), delimiter=';', encoding='utf-8',
                         dtype=str)
             for f in os.listdir(data_path)
             if f.startswith(file_start)]
    return pd.concat(files, ignore_index=True)


def patient_selection(registry_path:str, eds_path:str, 
                      exclude_patients_under_18:bool = True, exclude_non_ischemic_stroke:bool = True, exclude_non_acute_stroke: bool = True,
                      exclude_intra_hospital_stroke:bool = True,
                      verbose:bool=True):
    """
    Select patients from the registry and EDS data based on specific criteria.
    
    Arguments:
    - registry_path: Path to the registry CSV file.
    - eds_path: Path to the EDS CSV file. - # general consent is present for the extraction of 20221117
    - exclude_patients_under_18: If True, exclude patients under 18 years old.
    - exclude_non_ischemic_stroke: If True, exclude patients with non-ischemic stroke.
    - exclude_non_acute_stroke: If True, exclude patients with non-acute stroke.
    - verbose: If True, print additional information about the selection process.

    Returns:
    - registry_df: DataFrame containing the selected patients from the registry.
    - excluded_patients_df: DataFrame containing the number of excluded patients based on the criteria.
    """

    eds_df = pd.read_csv(eds_path, delimiter=';', encoding='utf-8',
                            dtype=str)
    registry_df = pd.read_excel(registry_path, dtype=str)

    eds_df['case_admission_id'] = create_ehr_case_identification_column(eds_df)
    registry_df['case_admission_id'] = create_registry_case_identification_column(registry_df)

    n_duplicates = len(registry_df[registry_df['Type of event'] == 'duplicate']['case_admission_id'].unique())
    n_duplicates += registry_df['case_admission_id'].duplicated().sum()
    n_records_screened = len(registry_df['case_admission_id'].unique()) - n_duplicates

    registry_df = registry_df[registry_df['Type of event'] != 'duplicate']
    registry_df = registry_df.drop_duplicates(subset='case_admission_id', keep='first')

    if verbose: 
        print('Number of records screened: ', n_records_screened, 'after removing duplicates: ', n_duplicates)

    excluded_patients_df = pd.DataFrame({
        'n_records_screened': n_records_screened,
    }, index=[0])

    # Eclude patients refusing to participate
    consent_df = registry_df[['case_admission_id', 'Patient refuses use of data for research']]\
        .merge(eds_df[['case_admission_id', 'patient_id_gc']], on='case_admission_id', how='left')

    patients_refusing_use_of_data = consent_df[(consent_df.patient_id_gc == 'Non') | ((consent_df['Patient refuses use of data for research'] == 'yes') & (consent_df.patient_id_gc != 'Oui'))].case_admission_id.unique()

    # Remove patients not wanting to participate in research
    n_patient_refuses_research = len(patients_refusing_use_of_data)
    if verbose:
        print(f'{n_patient_refuses_research} patients refuse the use of their data')
    registry_df = registry_df[~registry_df.case_admission_id.isin(patients_refusing_use_of_data)]

    excluded_patients_df['n_patient_refuses_research'] = n_patient_refuses_research

    if exclude_non_ischemic_stroke:
        # select only ischemic stroke patients
        n_patients_not_ischemic_stroke = len(registry_df['case_admission_id'].unique()) - len(registry_df[registry_df['Type of event'] == 'Ischemic stroke']['case_admission_id'].unique())
        registry_df = registry_df[registry_df['Type of event'] == 'Ischemic stroke']
        if verbose:
            print('Number of patients excluded because not ischemic stroke: ', n_patients_not_ischemic_stroke)
        excluded_patients_df['n_patients_not_ischemic_stroke'] = n_patients_not_ischemic_stroke

    if exclude_non_acute_stroke:
        # exclude non acute stroke patients
        datatime_format = '%d.%m.%Y %H:%M'
        registry_df['arrival_dt'] = pd.to_datetime(registry_df['Arrival at hospital'],
                                                        format='%Y%m%d').dt.strftime('%d.%m.%Y') + ' ' + \
                                        pd.to_datetime(registry_df['Arrival time'], format='%H:%M',
                                                        infer_datetime_format=True).dt.strftime('%H:%M')

        registry_df['stroke_dt'] = pd.to_datetime(registry_df['Onset date'],
                                                        format='%Y%m%d').dt.strftime('%d.%m.%Y') + ' ' + \
                                            pd.to_datetime(registry_df['Onset time'], format='%H:%M',
                                                        infer_datetime_format=True).dt.strftime('%H:%M')

        registry_df['delta_onset_arrival'] = (
                pd.to_datetime(registry_df['stroke_dt'], format=datatime_format, errors='coerce')
                - pd.to_datetime(registry_df['arrival_dt'], format=datatime_format, errors='coerce')
        ).dt.total_seconds()
        cid_with_non_acute_stroke = registry_df[registry_df['delta_onset_arrival'] < -7 * 24 * 60 * 60]['case_admission_id'].unique()
        n_with_non_acute_stroke = len(cid_with_non_acute_stroke)
        if verbose:
            print('Number of patients excluded because non acute stroke: ', n_with_non_acute_stroke)
        registry_df = registry_df[~registry_df['case_admission_id'].isin(cid_with_non_acute_stroke)]

        excluded_patients_df['n_with_non_acute_stroke'] = n_with_non_acute_stroke

    if exclude_patients_under_18:
        # find cids with patients under 18 (exclude patients with age < 1 as this represents wrong DOB)
        cids_under_18 = registry_df[(registry_df['Age (calc.)'].astype(float) < 18) & (registry_df['Age (calc.)'].astype(float) > 1)]['case_admission_id'].unique()
        n_patients_under_18 = len(cids_under_18)
        if verbose:
            print('Number of patients excluded because under 18: ', n_patients_under_18)
        registry_df = registry_df[~registry_df['case_admission_id'].isin(cids_under_18)]
        excluded_patients_df['n_patients_under_18'] = n_patients_under_18

    if exclude_intra_hospital_stroke:
        # find cids with intra-hospital stroke
        cids_intra_hospital_stroke = registry_df[registry_df['Referral'] == 'In-hospital event']['case_admission_id'].unique()
        n_patients_with_intra_hospital_stroke = len(cids_intra_hospital_stroke)
        if verbose: 
            print('Number of patients excluded because intra-hospital stroke: ', n_patients_with_intra_hospital_stroke)
        registry_df = registry_df[~registry_df['case_admission_id'].isin(cids_intra_hospital_stroke)]
        excluded_patients_df['n_patients_with_intra_hospital_stroke'] = n_patients_with_intra_hospital_stroke

    # excluded patients logs
    excluded_patients_df = excluded_patients_df.T
    excluded_patients_df.columns = ['number of patients']

    return registry_df, excluded_patients_df