from pathlib import Path
from subprocess import check_output

from click.testing import CliRunner
from mock import MagicMock, patch
from nose.tools import assert_equal, ok_

from neurom.apps.cli import cli

DATA = Path(__file__).parent.parent.parent.parent.resolve() / 'test_data'


@patch('neurom.apps.cli.plt.show')
def test_viewer_matplotlib(mock):
    runner = CliRunner()
    filename = str(DATA / 'swc' / 'simple.swc')

    result = runner.invoke(cli, ['view', filename])
    assert_equal(result.exit_code, 0)
    mock.assert_called_once()

    mock.reset_mock()
    result = runner.invoke(cli, ['view', filename, '--plane', 'xy'])
    assert_equal(result.exit_code, 0)
    mock.assert_called_once()


@patch('neurom.view.plotly.plot')
def test_viewer_plotly(mock):
    runner = CliRunner()
    filename = str(DATA / 'swc' / 'simple.swc')

    result = runner.invoke(cli, ['view', filename,
                                 '--backend', 'plotly'])
    assert_equal(result.exit_code, 0)
    mock.assert_called_once()

    mock.reset_mock()
    result = runner.invoke(cli, ['view', filename,
                                 '--backend', 'plotly',
                                 '--plane', 'xy'])
    assert_equal(result.exit_code, 0)
    mock.assert_called_once()

def test_morph_stat():
    check_output(['morph_stats', '-l'])
