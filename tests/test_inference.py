import sys

import torch
import torch.utils.model_zoo

from scripts.benchmark_nyu import NYUInputs, ROOT, build_model, model_config, summary, DepthOnly


def test_random_initialization_never_loads_weights(monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError('Random inference must not load or download weights')
    monkeypatch.setattr(torch, 'load', forbidden)
    monkeypatch.setattr(torch.hub, 'load_state_dict_from_url', forbidden)
    monkeypatch.setattr(torch.utils.model_zoo, 'load_url', forbidden)
    torch.set_num_threads(1)
    model = build_model(model_config(2023, ROOT / 'data/nyudepthv2_h5'))
    assert type(model).__name__ == 'PENet_C2'
    assert model.backbone.geoplanes == 3
    assert model.encoder3.device.type == 'cpu'
    assert 'cv2' not in sys.modules


def test_fixed_cspn_kernels_and_nyu_geometry_are_preserved():
    args = model_config(2023, ROOT / 'data/nyudepthv2_h5')
    model = build_model(args)
    for kernel, size in ((model.encoder3, 3), (model.encoder5, 5), (model.encoder7, 7)):
        assert not kernel.requires_grad
        assert (kernel == 1).sum() == size * size
        assert (kernel != 0).sum() == size * size
    wrapper = DepthOnly(model)
    assert wrapper.position.shape == (1, 2, 256, 320)
    assert wrapper.K.shape == (1, 3, 3)
    assert torch.isclose(wrapper.K[0, 0, 2], torch.tensor(313.04475870804731 / 2 - 8))
    assert 'position' not in wrapper.state_dict() and 'K' not in wrapper.state_dict()


def test_nyu_input_is_deterministic_and_has_500_points():
    dataset = NYUInputs(ROOT / 'data/nyudepthv2_h5')
    first, repeat = dataset[0], dataset[0]
    assert len(dataset) == 654
    assert first['rgb'].shape == (1, 3, 228, 304)
    assert first['dep'].shape == (1, 1, 228, 304)
    assert (first['dep'] > 0).sum() == 500
    assert torch.equal(first['dep'], repeat['dep'])


def test_summary_uses_all_measurements():
    result = summary([10, 20, 30])
    assert result['count'] == 3 and result['mean_ms'] == 20
    assert result['p95_ms'] == 29 and result['fps'] == 50


def test_geometry_opt_in_keeps_legacy_default():
    from types import SimpleNamespace
    from model import ENet
    args = SimpleNamespace(network_model='pe', dilation_rate=2,
                           convolutional_layer_encoding='xyz')
    model = ENet(args)
    assert not hasattr(model.args, 'geometry_size')
    adapted = model_config(2023, ROOT / 'data/nyudepthv2_h5')
    assert adapted.geometry_size == [256, 320]
    assert adapted.padding_lrtb == [0, 16, 0, 28]
