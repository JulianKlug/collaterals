import pandas as pd
import numpy as np
from pathlib import Path

from lab_preprocessing import preprocess_labs


def _base_lab_row(**overrides):
    base = {
        'nr': 0,
        'patient_id': '1',
        'eds_end_4digit': '1234',
        'eds_manual': '',
        'DOB': '',
        'begin_date': '',
        'end_date': '',
        'death_date': '',
        'death_hosp': '',
        'eds_final_id': '',
        'eds_final_begin': '',
        'eds_final_end': '',
        'eds_final_patient_id': '',
        'eds_final_birth': '',
        'eds_final_death': '',
        'eds_final_birth_str': '',
        'date_from': '',
        'date_to': '',
        'patient_id_manual': '',
        'stroke_onset_date': '',
        'Referral': '',
        'match_by': '',
        'multiple_id': '',
        'case_admission_id': '',
        'sample_date': '01.01.2020 12:00',
        'analyse_label': 'albumine analysis',
        'dosage_label': 'albumine',
        'material_label': 'sga',
        'unit_of_measure': 'g/l',
        'value': '10',
        'lower_limit': '0',
        'upper_limit': '100',
    }
    base.update(overrides)
    return base


def test_preprocess_labs_median_counts(tmp_path: Path):
    rows = [
        _base_lab_row(value='10'),
        _base_lab_row(value='20'),
        _base_lab_row(patient_id='2', eds_end_4digit='9999', value='30'),
    ]
    lab_df = pd.DataFrame(rows)

    processed = preprocess_labs(lab_df, selected_variables=['albumine'], log_dir=str(tmp_path), verbose=False)

    # Returned dataframe should have numeric values and only the selected dosage
    assert processed['dosage_label'].unique().tolist() == ['albumine']
    assert processed['value'].dtype.kind in {'i', 'u', 'f'}

    # Median observations per case_admission_id is written when log_dir is provided
    median_path = tmp_path / 'median_observations_per_case_admission_id.csv'
    median_df = pd.read_csv(median_path)

    # Expected median: counts per case are [2, 1] -> median = 1.5
    expected = pd.DataFrame({'dosage_label': ['albumine'], 'median_observations_per_case_admission_id': [1.5]})
    pd.testing.assert_frame_equal(median_df.sort_index(axis=1), expected.sort_index(axis=1))

    # Ensure no object dtype sneaks into median calculation
    assert median_df['median_observations_per_case_admission_id'].dtype.kind in {'i', 'u', 'f'}
