from prometheus.intake.profiler import DataProfiler


if __name__ == "__main__":
    profiler = DataProfiler("tests/fixtures/sample.csv")
    print(profiler.profile(target="survived"))
