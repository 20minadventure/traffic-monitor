import numpy as np
import pytest

from traffic_detection import patch_generator


@pytest.mark.parametrize(
    "patch_generator_kwargs, patches_out",
    [
        (
            {'patch_size': 1, 'patches': (3, 4)},
            [[[i]] for i in range(12)]
        ),
        (
            {'patch_size': 3, 'patches': (1, 1)},
            []
        ),
        (
            {'patch_size': 3, 'patches': (1, 1), 'start_point': (1, 0)},
            [[[j for j in range(i, i + 3)] for i in range(1, 12, 4)]]
        ),
        (
            {'patch_size': 3, 'patches': (1, 1), 'start_point': (0, 1)},
            []
        ),
        (
            {'patch_size': 1, 'start_point': (1, 0), 'end_point': (4, 1)},
            [[[1]], [[2]], [[3]]]
        ),
        (
            {'patch_size': 4, 'patches': (1, 1)},
            []
        ),
        (
            {'patch_size': 1, 'min_padding': 1},
            []
        ),
        (
            {'patch_size': 3, 'min_padding': 1},
            [[[k for k in range(j, j + 3)] for j in range(i, 12, 4)] for i in range(2)]
        ),
        (
            {'patch_size': 20, 'min_padding': 10},
            []
        ),
        (
            {'patch_size': 2, 'patches': (2, 2)},
            [[[j, j + 1] for j in [i, i + 4]] for i in range(0, 7, 2)]
        ),
    ]
)
def test_patch_generation_2d(patch_generator_kwargs, patches_out):
    image = np.arange(12).reshape(3, 4).astype(np.uint8)

    patches = [
        patch.tolist()
        for patch, _ in patch_generator(image, **patch_generator_kwargs)
    ]
    
    assert patches == patches_out
