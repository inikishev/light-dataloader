import pytest
import torch

@pytest.mark.parametrize("batch_size", [2, 32, 128])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("memory_efficient", [True, False])
@pytest.mark.parametrize("seed", [0, None])
def test_tensordataloader(batch_size, shuffle, memory_efficient, seed):
    from lightdl import TensorDataLoader

    data = torch.arange(0, 1000)[:, None,None,None].repeat_interleave(3, 1).repeat_interleave(32, 2).repeat_interleave(32, 3)
    labels = torch.arange(0, 1000)
    seen = set()
    dataloader = TensorDataLoader((data, labels), batch_size=batch_size, shuffle=shuffle, memory_efficient=memory_efficient, seed=seed)

    for i, batch in enumerate(dataloader):
        if i == len(dataloader) - 1:
            expected_size = len(data) - (batch_size * (len(dataloader)-1))

        else:
            expected_size = batch_size

        assert batch[0].shape == (expected_size, 3, 32, 32), batch[0].shape
        assert batch[1].shape == (expected_size,), batch[1].shape

        value_x = batch[0][:,0,0,0]
        value_y = batch[1]

        assert (value_x == value_y).all()
        assert len(seen.intersection(value_y)) == 0, seen.intersection(value_y)
        seen = seen.union(value_x.tolist())

    assert seen == set(range(1000)), set(range(1000)).difference(seen)


@pytest.mark.parametrize("batch_size", [2, 32, 128])
@pytest.mark.parametrize("shuffle", [True, False])
@pytest.mark.parametrize("seed", [0, None])
def test_lightdataloader(batch_size, shuffle, seed):
    from lightdl import LightDataLoader

    data = torch.arange(0, 1000)[:, None,None,None].repeat_interleave(3, 1).repeat_interleave(32, 2).repeat_interleave(32, 3)
    labels = torch.arange(0, 1000)
    seen = set()
    dataloader = LightDataLoader(list(zip(data, labels)), batch_size=batch_size, shuffle=shuffle, seed=seed)

    for i, batch in enumerate(dataloader):
        if i == len(dataloader) - 1:
            expected_size = len(data) - (batch_size * (len(dataloader)-1))

        else:
            expected_size = batch_size

        assert batch[0].shape == (expected_size, 3, 32, 32), batch[0].shape
        assert batch[1].shape == (expected_size,), batch[1].shape

        value_x = batch[0][:,0,0,0]
        value_y = batch[1]

        assert (value_x == value_y).all()
        assert len(seen.intersection(value_y)) == 0, seen.intersection(value_y)
        seen = seen.union(value_x.tolist())

    assert seen == set(range(1000)), set(range(1000)).difference(seen)