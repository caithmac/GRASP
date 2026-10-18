import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import torch
from safetensors.torch import save_file

from DeBERTa.deberta.config import ModelConfig
from grasp import GRASPEncoder, GRASPPredictor
from grasp.encoder import encoder_inputs, make_loader, read_vocab
from grasp.predictor import PropertyNetwork
from grasp.train import fit
from mole.training.models.mole import AtomEnvEmbeddings


@pytest.fixture
def encoder():
    root = Path(__file__).resolve().parents[1]
    config = json.loads((root/'hf_model/config.json').read_text())
    config['architecture'].update(hidden_size=24, embedding_size=24, intermediate_size=48,
                                  num_attention_heads=4, attention_head_size=6)
    torch.manual_seed(42)
    model = AtomEnvEmbeddings(ModelConfig.from_dict(config['architecture']))
    return GRASPEncoder(model, read_vocab(root/'hf_model/vocab.json'), config, 'fixture', 'cpu')


def test_final_readout_uses_final_atoms_and_preserves_head_initialization(encoder):
    torch.manual_seed(17)
    mixed = GRASPPredictor.from_encoder(encoder, task='regression', method='frozen')
    torch.manual_seed(17)
    final = GRASPPredictor.from_encoder(encoder, task='regression', method='frozen', mix_layers=(12,))
    for key, tensor in mixed.model.state_dict().items():
        if key.startswith(('head.', 'pooler.')):
            torch.testing.assert_close(tensor, final.model.state_dict()[key], rtol=0, atol=0)
    batch = next(iter(make_loader(['CCO', 'c1ccccc1'], encoder.vocabulary, 2)))
    final.model.eval()
    captured = []
    hook = final.model.pooler.register_forward_pre_hook(lambda _, args: captured.append(args[0].detach()))
    with torch.no_grad():
        final.model(batch)
        ids, mask, distances = encoder_inputs(batch)
        expected = final.model.encoder(ids, mask, attention_mask=mask, relative_pos=distances)['hidden_states'][-1]
    hook.remove()
    torch.testing.assert_close(captured[0], expected, rtol=0, atol=0)


@pytest.mark.parametrize('method', ['frozen', 'full'])
def test_gradient_policy_and_readout_reload(encoder, tmp_path, method):
    predictor = GRASPPredictor.from_encoder(encoder, task='regression', method=method, mix_layers=(12,))
    batch = next(iter(make_loader(['CCO', 'CCN'], encoder.vocabulary, 2)))
    predictor.model.train()
    predictor.model(batch).sum().backward()
    gradients = [p.grad for p in predictor.model.encoder.parameters()]
    if method == 'frozen':
        assert all(g is None for g in gradients)
    else:
        assert any(g is not None and torch.count_nonzero(g) for g in gradients)
    assert predictor.model.head[-1].weight.grad is not None
    predictor.model.eval()
    before = predictor.predict(['CCO', 'CCN'])
    predictor.save_pretrained(tmp_path)
    restored = GRASPPredictor.from_pretrained(tmp_path, device='cpu')
    assert restored.model.mix_layers == (12,)
    np.testing.assert_array_equal(before, restored.predict(['CCO', 'CCN']))


def test_old_predictor_config_still_loads(encoder, tmp_path):
    predictor = GRASPPredictor.from_encoder(encoder, task='regression', method='frozen')
    predictor.save_pretrained(tmp_path)
    config = json.loads((tmp_path/'predictor_config.json').read_text())
    config.pop('mix_layers')
    (tmp_path/'predictor_config.json').write_text(json.dumps(config))
    restored = GRASPPredictor.from_pretrained(tmp_path, device='cpu')
    assert restored.model.mix_layers == (4, 8, 10, 12)
    np.testing.assert_array_equal(predictor.predict(['CCO']), restored.predict(['CCO']))


@pytest.mark.parametrize('layers', [(), (0,), (13,), (12, 12), (12.0,), (True,)])
def test_invalid_readout_is_rejected(encoder, layers):
    with pytest.raises(ValueError, match='mix_layers'):
        PropertyNetwork(encoder.config['architecture'], 'full', mix_layers=layers)


def test_training_passes_readout_to_saved_predictor(encoder, tmp_path):
    source = tmp_path/'encoder'
    source.mkdir()
    save_file(encoder.model.state_dict(), source/'model.safetensors')
    (source/'config.json').write_text(json.dumps(encoder.config))
    (source/'vocab.json').write_text(json.dumps(encoder.vocabulary))
    train, valid = tmp_path/'train.csv', tmp_path/'valid.csv'
    pd.DataFrame({'smiles':['CCO', 'CCN', 'CCC', 'CC(=O)O'], 'target':[0., 1., 2., 3.]}).to_csv(train, index=False)
    pd.DataFrame({'smiles':['c1ccccc1', 'CCCl'], 'target':[1., 2.]}).to_csv(valid, index=False)
    fit(model_source=source, train_csv=train, valid_csv=valid, output_dir=tmp_path/'fit',
        task='regression', method='frozen', mix_layers=(12,), epochs=1, patience=1, batch_size=2, device='cpu')
    restored = GRASPPredictor.from_pretrained(tmp_path/'fit', device='cpu')
    assert restored.config['mix_layers'] == [12]
    assert np.isfinite(restored.predict(['CO'])).all()
