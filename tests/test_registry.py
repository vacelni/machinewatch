import pytest

from machinewatch.registry import Registry


class Constant:
    def __init__(self, value):
        self.value = value


def test_a_worse_model_is_registered_but_not_promoted(tmp_path):
    registry = Registry(tmp_path)
    good = registry.register(Constant(1), "good", {"rmse": 14.0})
    assert registry.promote_if_better(good)
    worse = registry.register(Constant(2), "worse", {"rmse": 20.0})
    assert not registry.promote_if_better(worse)
    assert registry.current_version() == good.version
    assert len(registry.versions()) == 2


def test_a_better_model_replaces_the_current_one(tmp_path):
    registry = Registry(tmp_path)
    registry.promote_if_better(registry.register(Constant(1), "a", {"rmse": 20.0}))
    better = registry.register(Constant(2), "b", {"rmse": 13.0})
    assert registry.promote_if_better(better)
    assert registry.load().value == 2


def test_a_tampered_model_file_is_refused(tmp_path):
    registry = Registry(tmp_path)
    meta = registry.register(Constant(1), "a", {"rmse": 1.0})
    registry.promote_if_better(meta)
    (tmp_path / meta.version / "model.joblib").write_bytes(b"not the model")
    with pytest.raises(ValueError, match="checksum"):
        registry.load()
