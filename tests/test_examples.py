import runpy
import shutil
from pathlib import Path

import pytest

EXAMPLES_DIR = Path(__file__).parent.parent / 'examples'
EXAMPLES = sorted(EXAMPLES_DIR.glob('*.py'))

# Optional dependencies each example needs beyond the base install
REQUIRES = {
    '4_etr_pandas.py': ['matplotlib', 'lmfit'],
    '5_realistic_example.py': ['matplotlib', 'lmfit'],
}


@pytest.mark.slow
@pytest.mark.parametrize('script', EXAMPLES, ids=lambda p: p.name)
def test_example_runs(script, tmp_path, monkeypatch):
    for module in REQUIRES.get(script.name, []):
        pytest.importorskip(module)

    monkeypatch.setenv('MPLBACKEND', 'Agg')
    try:
        import matplotlib
        import matplotlib.pyplot as plt
    except ImportError:
        plt = None
    else:
        matplotlib.use('Agg')

    # Run a copy so outputs written next to the script stay out of the repo
    workdir = tmp_path / 'examples'
    shutil.copytree(EXAMPLES_DIR, workdir,
                    ignore=shutil.ignore_patterns('results', '__pycache__'))
    monkeypatch.chdir(workdir)

    try:
        runpy.run_path(str(workdir / script.name), run_name='__main__')
    finally:
        if plt is not None:
            plt.close('all')


def test_readme_quickstart_matches_example():
    # README is rendered on GitHub/PyPI, which can't resolve snippet includes,
    # so its quickstart is a copy of the marked region in 0_quickstart.py
    script = (EXAMPLES_DIR / '0_quickstart.py').read_text()
    start = script.index('# --8<-- [start:quickstart]\n') + len('# --8<-- [start:quickstart]\n')
    end = script.index('# --8<-- [end:quickstart]')
    snippet = script[start:end]

    readme = (EXAMPLES_DIR.parent / 'README.md').read_text()
    assert f'```python\n{snippet}```' in readme
