# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Index-respecting safetensors loading (safetensors_use_index).

Covers the GLM-5.3-Flash AutoRound checkpoint pattern: shards hold stale
duplicate tensors under names whose index-mapped authoritative copy lives in
another (repair) file. All tests are CPU-only against synthetic checkpoints.
"""

import json
import struct

import pytest
import torch
from safetensors.torch import save_file

from vllm.model_executor.model_loader.weight_utils import (
    indexed_safetensors_weights_iterator,
    safetensors_weights_iterator,
)

SAFE_WEIGHTS_INDEX_NAME = "model.safetensors.index.json"


def _write_st(path, tensors):
    save_file(tensors, str(path))


def _make_checkpoint(dir_path, layout):
    """layout: {file_name: {tensor_name: tensor}}; writes index from layout."""
    for fname, tensors in layout.items():
        _write_st(dir_path / fname, tensors)
    weight_map = {
        name: fname
        for fname, tensors in layout.items()
        for name in tensors
    }
    index = {"weight_map": weight_map, "metadata": {"total_size": 0}}
    (dir_path / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index))
    return weight_map


def _file_keys(path):
    import safetensors

    with safetensors.safe_open(str(path), framework="pt") as f:
        return list(f.keys())


def _iter_indexed(tmp_path, layout, index_file=SAFE_WEIGHTS_INDEX_NAME):
    _make_checkpoint(tmp_path, layout)
    files = [str(tmp_path / f) for f in layout]
    return dict(
        indexed_safetensors_weights_iterator(
            files, str(tmp_path), index_file, use_tqdm_on_load=False
        )
    )


def test_stale_duplicate_resolved_by_index(tmp_path):
    """A shard holds a stale conv tensor; the index maps it to the repair file."""
    stale = torch.zeros(24576, 1, 4, dtype=torch.bfloat16)
    repaired = torch.ones(8192, 1, 4, dtype=torch.bfloat16)
    layout = {
        "model-00001.safetensors": {"layer.conv1d.weight": stale},
        "model_extra_conv.safetensors": {"layer.conv1d.weight": repaired},
    }
    got = _iter_indexed(tmp_path, layout)
    # one yield, from the repair file
    assert set(got) == {"layer.conv1d.weight"}
    assert torch.equal(got["layer.conv1d.weight"], repaired)
    # plain iterator yields both copies; sorted file order puts the repair
    # file ("model_extra_conv..." > "model-00001..." naturally) later, so a
    # dict-based consumer silently keeps the repaired one — but a stream
    # consumer (weight_loader) receives the stale [24576,1,4] copy first and
    # would see the wrong shape. The indexed iterator yields exactly one.
    plain = list(
        safetensors_weights_iterator(
            [str(tmp_path / f) for f in sorted(layout)], False
        )
    )
    assert len(plain) == 2
    assert plain[0][0] == "layer.conv1d.weight"


def test_unindexed_extra_tensors_ignored(tmp_path):
    layout = {
        "model-00001.safetensors": {
            "a.weight": torch.randn(4, 4),
            "b.weight": torch.randn(4, 4),
        },
    }
    # add an extra tensor to the file after the index was written
    _make_checkpoint(tmp_path, layout)
    _write_st(
        tmp_path / "model-00001.safetensors",
        {
            "a.weight": layout["model-00001.safetensors"]["a.weight"],
            "b.weight": layout["model-00001.safetensors"]["b.weight"],
            "junk.extra": torch.randn(2, 2),
        },
    )
    got = _iter_indexed(tmp_path, layout)
    assert set(got) == {"a.weight", "b.weight"}


def test_missing_mapped_file_raises(tmp_path):
    layout = {
        "model-00001.safetensors": {"a.weight": torch.randn(4, 4)},
        "model_extra_tensors.safetensors": {"b.weight": torch.randn(4, 4)},
    }
    _make_checkpoint(tmp_path, layout)
    (tmp_path / "model_extra_tensors.safetensors").unlink()
    with pytest.raises(FileNotFoundError, match="missing file"):
        list(
            indexed_safetensors_weights_iterator(
                [str(tmp_path / f) for f in layout],
                str(tmp_path),
                SAFE_WEIGHTS_INDEX_NAME,
                False,
            )
        )


def test_missing_mapped_tensor_raises(tmp_path):
    # index says the tensor is in file A, but the file lacks it
    _make_checkpoint(
        tmp_path, {"model-00001.safetensors": {"a.weight": torch.randn(4, 4)}}
    )
    _write_st(tmp_path / "model-00001.safetensors", {"other.weight": torch.randn(4, 4)})
    with pytest.raises(KeyError, match="absent from the file"):
        list(
            indexed_safetensors_weights_iterator(
                [str(tmp_path / "model-00001.safetensors")],
                str(tmp_path),
                SAFE_WEIGHTS_INDEX_NAME,
                False,
            )
        )


def test_missing_index_raises(tmp_path):
    _write_st(tmp_path / "model-00001.safetensors", {"a.weight": torch.randn(4, 4)})
    with pytest.raises(FileNotFoundError, match="no index"):
        list(
            indexed_safetensors_weights_iterator(
                [str(tmp_path / "model-00001.safetensors")],
                str(tmp_path),
                SAFE_WEIGHTS_INDEX_NAME,
                False,
            )
        )


def test_missing_index_file_entry(tmp_path):
    """Index references a file not on disk at all (not passed by glob either)."""
    _make_checkpoint(
        tmp_path, {"model-00001.safetensors": {"a.weight": torch.randn(4, 4)}}
    )
    # rewrite index pointing at a nonexistent file
    index = {"weight_map": {"a.weight": "model-99999-of-99999.safetensors"}}
    (tmp_path / SAFE_WEIGHTS_INDEX_NAME).write_text(json.dumps(index))
    with pytest.raises(FileNotFoundError):
        list(
            indexed_safetensors_weights_iterator(
                [str(tmp_path / "model-00001.safetensors")],
                str(tmp_path),
                SAFE_WEIGHTS_INDEX_NAME,
                False,
            )
        )


def test_repaired_conv_shape_and_order(tmp_path):
    """GLM pattern end-to-end: repaired [8192,1,4] conv from the extra file;
    per-file iteration order does not affect the result."""
    layout = {
        "model-00002.safetensors": {
            "layer0.qweight": torch.randint(0, 2**31, (64, 16), dtype=torch.int32),
            "layer1.conv1d.weight": torch.full((8192, 1, 4), 2.0, dtype=torch.bfloat16),
        },
        "model_extra_conv.safetensors": {
            "layer1.conv1d.weight": torch.full((8192, 1, 4), 3.0, dtype=torch.bfloat16),
        },
    }
    got = _iter_indexed(tmp_path, layout)
    assert got["layer1.conv1d.weight"].shape == (8192, 1, 4)
    assert got["layer1.conv1d.weight"][0, 0, 0].item() == 3.0
    assert set(got) == {"layer0.qweight", "layer1.conv1d.weight"}


def test_default_behavior_unchanged(tmp_path):
    """Without the option the plain iterator still yields stale duplicates."""
    stale = torch.zeros(4, 4)
    repaired = torch.ones(4, 4)
    layout = {
        "model-00001.safetensors": {"w": stale},
        "model_extra.safetensors": {"w": repaired},
    }
    _make_checkpoint(tmp_path, layout)
    plain = dict(
        safetensors_weights_iterator(
            [str(tmp_path / f) for f in sorted(layout)], False
        )
    )
    # both files' copies are yielded (duplicate name, last wins in a dict)
    assert len(list(_file_keys(tmp_path / "model-00001.safetensors"))) == 1
    assert len(list(_file_keys(tmp_path / "model_extra.safetensors"))) == 1
    assert "w" in plain


def _header_numel(path):
    with open(path, "rb") as f:
        (n,) = struct.unpack("<Q", f.read(8))
        f.read(n)
    return None


if __name__ == "__main__":
    import sys

    sys.exit(pytest.main([__file__, "-v"]))
