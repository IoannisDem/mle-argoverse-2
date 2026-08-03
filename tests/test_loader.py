import pytest
import numpy as np
import loader
import numpy as np
import pytest

import loader


class TestGetDatapoints:
    TESTCASES = (
        pytest.param(
            {
                "frame_sequence": [1, 2, 3, 4, 5],
                "window_size": 3,
                "stride": 1,
                "expected": [
                    ([1, 2, 3], 4),
                    ([2, 3, 4], 5),
                ],
            },
        ),
        pytest.param(
            {
                "frame_sequence": [1, 2, 3, 4, 5],
                "window_size": 2,
                "stride": 2,
                "expected": [
                    ([1, 2], 3),
                    ([3, 4], 5),
                ],
            },
        ),
        pytest.param(
            {
                "frame_sequence": [1, 2, 3],
                "window_size": 3,
                "stride": 1,
                "expected": [],
            },
        ),
        pytest.param(
            {
                "frame_sequence": [],
                "window_size": 3,
                "stride": 1,
                "expected": [],
            },
        ),
        pytest.param(
            {
                "frame_sequence": [1, 2, 3, 4, 5, 6, 7],
                "window_size": 2,
                "stride": 3,
                "expected": [
                    ([1, 2], 3),
                    ([4, 5], 6),
                ],
            },
        ),
    )

    @pytest.fixture(params=TESTCASES)
    def test_case(self, request):
        return request.param

    def _make_frame_spec(self, frames):
        frame_array = np.array(frames)
        current_state = np.array([0])
        next_action = np.array([0])
        return loader.FrameSpec(
            frame_array,
            current_state,
            next_action,
        )

    @pytest.fixture
    def episode(self, test_case):
        frames_spec = [self._make_frame_spec(i) for i in test_case["frame_sequence"]]
        return loader.EpisodeSequence(
            episode_name="foo",
            frame_sequence=frames_spec,
        )

    @pytest.fixture
    def expected(self, test_case):
        frame_points = []
        for frame_values, label_value in test_case["expected"]:
            frame_points.append(
                loader.FrameDatapoint(
                    frames=[self._make_frame_spec(i) for i in frame_values],
                    label=self._make_frame_spec(label_value),
                )
            )

        return frame_points

    def test_get_datapoints(self, test_case, episode, expected):
        observed = loader.get_datapoints(
            episode,
            window_size=test_case["window_size"],
            stride=test_case["stride"],
        )

        assert observed == expected
