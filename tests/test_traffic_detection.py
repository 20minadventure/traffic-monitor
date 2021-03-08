from traffic_detection import TrafficDetector


def test_dry_run_prediction(tmp_path):
    clip_path = tmp_path / 'clip.mp4'

    td = TrafficDetector(clip_path)
    td.detect_vehicles()

    assert hasattr(td, 'confs')
    assert hasattr(td, 'boxes')
    assert hasattr(td, 'class_names')
