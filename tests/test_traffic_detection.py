import pytest


@pytest.fixture
def dummy_clip_path(tmp_path):
    return tmp_path / 'dummy_clip.mp4'
