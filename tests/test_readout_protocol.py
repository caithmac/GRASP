import importlib.util
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def protocol():
    path = Path(__file__).resolve().parents[1]/'notebooks/readout_helpers.py'
    spec = importlib.util.spec_from_file_location('readout_protocol_test', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_random_selection_is_fixed_and_without_replacement(protocol):
    expected = ['asap_sars', 'biogen_rlm', 'asap_logd', 'expansion_caco2_pappa',
                'expansion_ksol', 'expansion_logd']
    assert protocol.selected_datasets() == expected
    assert len(protocol.DATASETS) == 23
    assert len(set(expected)) == 6


def test_splits_keep_repeated_connectivity_and_refuse_changed_seed(protocol, tmp_path):
    frame = pd.DataFrame([dict(source_row=i, connectivity=f'mol_{i//2}',
                              scaffold=f'scaffold_{i//2}', smiles='CCO', target=float(i)) for i in range(40)])
    manifest = protocol.save_splits(frame, 'regression', tmp_path, 42)
    before = {key:protocol.sha256(tmp_path/f'{key}.csv') for key in manifest['sha256']}
    with pytest.raises(AssertionError, match='Existing splits differ'):
        protocol.save_splits(frame, 'regression', tmp_path, 43)
    assert before == {key:protocol.sha256(tmp_path/f'{key}.csv') for key in before}


def test_completed_cell_rejects_wrong_protocol_and_changed_predictions(protocol, tmp_path):
    plan = {'readout':[12], 'seed':0}
    protocol.write_json(tmp_path/'run_plan.json', plan)
    (tmp_path/'test_predictions.csv').write_text('target,prediction\n1,1\n')
    result = {'prediction_sha256':protocol.sha256(tmp_path/'test_predictions.csv')}
    protocol.write_json(tmp_path/'FIT_COMPLETE.json', result)
    assert protocol.validate_cached_cell(tmp_path, plan) == result
    with pytest.raises(ValueError, match='Settings'):
        protocol.validate_cached_cell(tmp_path, {'readout':[4, 8, 10, 12], 'seed':0})
    (tmp_path/'test_predictions.csv').write_text('target,prediction\n1,2\n')
    with pytest.raises(ValueError, match='predictions changed'):
        protocol.validate_cached_cell(tmp_path, plan)


def test_comparison_direction_and_missing_arm(protocol):
    rows = [dict(dataset='toy', seed=0, arm=arm, mae=value) for arm, value in
            [('frozen_final', 1), ('frozen_mixed', 2), ('full_final', 3), ('full_mixed', 1)]]
    paired, _ = protocol.comparison_tables(rows)
    values = paired.set_index('adaptation').final_minus_mixed
    assert values['frozen'] == -1 and values['full'] == 2
    with pytest.raises(ValueError, match='Missing arm'):
        protocol.comparison_tables(rows[:-1])
