import numpy as np
import pytest

from src.models.x_vla import validate_chunk


def test_validates_the_wire_shape_and_keeps_the_auxiliary_block():
    raw = np.zeros((30, 20))
    raw[:, 3] = raw[:, 7] = 1
    raw[:, 9] = 0.5
    raw[:, 19] = 0.8
    np.testing.assert_array_equal(validate_chunk(raw.tolist()), raw)


@pytest.mark.parametrize(
    "column, value, message",
    [(3, 0, "degenerate EEF6D"), (9, 1.01, "probability"), (19, np.nan, "non-finite")],
)
def test_rejects_invalid_output_before_decoding(column, value, message):
    raw = np.zeros((30, 20))
    raw[:, 3] = raw[:, 7] = 1
    raw[17, column] = value
    with pytest.raises(ValueError, match=message):
        validate_chunk(raw)


def test_rejects_a_truncated_or_already_decoded_chunk():
    with pytest.raises(ValueError, match="numeric X-VLA action chunk"):
        validate_chunk(np.zeros((30, 10)))
