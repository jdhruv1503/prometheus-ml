from prometheus.intake.profiler import DataProfiler


def test_profiler_outputs_shape_and_target_distribution():
    profiler = DataProfiler("tests/fixtures/sample.csv")
    profile = profiler.profile(target="survived")
    assert profile["shape"]["rows"] == 100
    assert "target_distribution" in profile
    assert "temporal_columns" in profile
