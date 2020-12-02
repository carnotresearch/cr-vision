
from cr.vision.dl.nets.cnn.utils import *
import pytest

def test_data_format():
    check_data_format("channels_first")
    check_data_format("channels_last")
    with pytest.raises(ValueError):
        check_data_format("channels_second")
    with pytest.raises(ValueError):
        check_data_format(None)


def test_check_3d_input():
    check_3d_input((3, 224, 224))
    check_3d_input((224, 224, 3))
    with pytest.raises(ValueError):
        check_3d_input((224, 224))
    with pytest.raises(ValueError):
        check_3d_input((20, 224, 224, 3))


def test_check_3_color_channels():
    check_3_color_channels((3, 224, 224), 'channels_first')
    with pytest.raises(ValueError):
        check_3_color_channels((3, 224, 224), 'channels_last')
    check_3_color_channels((224, 224, 3), 'channels_last')
    with pytest.raises(ValueError):
        check_3_color_channels((224, 224, 3), 'channels_first')
    with pytest.raises(ValueError):
        check_3_color_channels((224, 224))
    with pytest.raises(ValueError):
        check_3_color_channels((20, 3, 224, 224))
    with pytest.raises(ValueError):
        check_3_color_channels((20, 224, 224, 3))
    check_3_color_channels((224, 224, 3))
    with pytest.raises(ValueError):
        check_3_color_channels((3, 224, 224))
    with pytest.raises(ValueError):
        check_3_color_channels((224, 224, 4))
    with pytest.raises(ValueError):
        check_3_color_channels((4, 224, 224), 'channels_first')

def test_get_num_channels():
    assert 1 == get_num_channels((224, 224)) 
    assert 3 == get_num_channels((224, 224, 3))
    assert 3 == get_num_channels((3, 224, 224), 'channels_first')
    assert 3 == get_num_channels((224, 224, 3), 'channels_last')

def test_check_min_size():
    check_min_size((None, None), 32)
    check_min_size((None, 224), 32)
    check_min_size((224, None), 32)
    check_min_size((224, 224), 32)
    with pytest.raises(ValueError):
        check_min_size((None, 16), 32)
    with pytest.raises(ValueError):
        check_min_size((16, None), 32)
    with pytest.raises(ValueError):
        check_min_size((16, 16), 32)
    with pytest.raises(ValueError):
        check_min_size((16, 224), 32)
    with pytest.raises(ValueError):
        check_min_size((224, 16), 32)
    # with channels last
    check_min_size((None, None, 3), 32)
    check_min_size((None, 224, 3), 32)
    check_min_size((224, None, 3), 32)
    check_min_size((224, 224, 3), 32)
    with pytest.raises(ValueError):
        check_min_size((None, 16, 3), 32)
    with pytest.raises(ValueError):
        check_min_size((16, None, 1), 32)
    with pytest.raises(ValueError):
        check_min_size((16, 16, 2), 32)
    with pytest.raises(ValueError):
        check_min_size((16, 224, 4), 32)
    with pytest.raises(ValueError):
        check_min_size((224, 16, 5), 32)
    # with channels first
    check_min_size((3, None, None), 32, data_format="channels_first")
    check_min_size((2, None, 224), 32, data_format="channels_first")
    check_min_size((1, 224, None), 32, data_format="channels_first")
    check_min_size((4, 224, 224), 32, data_format="channels_first")
    with pytest.raises(ValueError):
        check_min_size((None, None, 16), 32, data_format="channels_first")
    with pytest.raises(ValueError):
        check_min_size((1, 16, None), 32, data_format="channels_first")
    with pytest.raises(ValueError):
        check_min_size((20, 16, 16), 32, data_format="channels_first")
    with pytest.raises(ValueError):
        check_min_size((4, 16, 224), 32, data_format="channels_first")
    with pytest.raises(ValueError):
        check_min_size((3, 224, 16), 32, data_format="channels_first")

def test_deduce_input_shape():
    assert (224, 224, 3) == deduce_input_shape()
    assert (256, 256, 3) == deduce_input_shape(default_size=256)
    assert (128, 128, 1) == deduce_input_shape((128, 128, 1))
    assert (128, 128, 3) == deduce_input_shape((128, 128, 3), weights="imagenet")
    assert (3, 128, 128) == deduce_input_shape((3, 128, 128), weights="imagenet", data_format="channels_first")
    # no flattening at the top
    assert (None, None, 1) == deduce_input_shape((None, None, 1), require_flatten=False)
    with pytest.warns(UserWarning):
        assert (None, None, 2) == deduce_input_shape((None, None, 2), require_flatten=False)
    assert (None, None, 3) == deduce_input_shape((None, None, 3), require_flatten=False)
    assert (None, None, 3) == deduce_input_shape((None, None, 3), weights="imagenet", require_flatten=False)
    with pytest.raises(ValueError):
        deduce_input_shape((16, 16, 3))
    with pytest.raises(ValueError):
        deduce_input_shape((None, 16, 3))
    with pytest.raises(ValueError):
        deduce_input_shape((16, None, 3))
    with pytest.raises(ValueError):
        deduce_input_shape((None, None, 3))
    pass