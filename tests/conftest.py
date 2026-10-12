import matplotlib
import pytest

# No display in CI: plots have to be rendered on a headless backend
matplotlib.use("Agg")


@pytest.fixture(autouse=True)
def close_figures():
    yield
    matplotlib.pyplot.close("all")
