"""The independent audit must not hide duplicate or missing events."""
from scripts.audit_libero_mem_writer import match_events


def test_event_matching_is_one_to_one():
    result = match_events([100, 101, 210], [100, 200], tolerance=20)
    assert result["true_positive"] == 2
    assert result["false_positive"] == 1
    assert result["false_negative"] == 0


def test_out_of_window_event_does_not_match():
    result = match_events([121], [100, 200], tolerance=20)
    assert result["true_positive"] == 0
    assert result["false_positive"] == 1
    assert result["false_negative"] == 2
