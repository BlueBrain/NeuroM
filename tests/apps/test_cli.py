import json
from pathlib import Path
import tempfile

import pandas as pd
import yaml
from click.testing import CliRunner
from mock import patch

from neurom.apps.cli import cli
from neurom.exceptions import ConfigError

DATA = Path(__file__).parent.parent / 'data'


@patch('neurom.apps.cli.plt.show')
def test_viewer_matplotlib(mock):
    runner = CliRunner()
    filename = str(DATA / 'swc' / 'simple.swc')

    result = runner.invoke(cli, ['view', filename])
    assert result.exit_code == 0
    mock.assert_called_once()

    mock.reset_mock()
    result = runner.invoke(cli, ['view', filename, '--plane', 'xy'])
    assert result.exit_code == 0
    mock.assert_called_once()


@patch('neurom.view.plotly.plot')
def test_viewer_plotly(mock):
    runner = CliRunner()
    filename = str(DATA / 'swc' / 'simple.swc')

    result = runner.invoke(cli, ['view', filename,
                                 '--backend', 'plotly'])
    assert result.exit_code == 0
    mock.assert_called_once()

    mock.reset_mock()
    result = runner.invoke(cli, ['view', filename,
                                 '--backend', 'plotly',
                                 '--plane', 'xy'])
    assert result.exit_code == 0
    mock.assert_called_once()


def test_morph_stat():
    runner = CliRunner()
    filename = DATA / 'swc' / 'simple.swc'
    with tempfile.NamedTemporaryFile() as f:
        result = runner.invoke(cli, ['stats', str(filename), '--output', f.name])
        assert result.exit_code == 0
        df = pd.read_csv(f)
        assert set(df.columns) == {'name', 'axon:max_section_length', 'axon:total_section_length',
                                   'axon:total_section_volume', 'axon:max_section_branch_order',
                                   'apical_dendrite:max_section_length',
                                   'apical_dendrite:total_section_length',
                                   'apical_dendrite:total_section_volume',
                                   'apical_dendrite:max_section_branch_order',
                                   'basal_dendrite:max_section_length',
                                   'basal_dendrite:total_section_length',
                                   'basal_dendrite:total_section_volume',
                                   'basal_dendrite:max_section_branch_order',
                                   'all:max_section_length',
                                   'all:total_section_length', 'all:total_section_volume',
                                   'all:max_section_branch_order', 'neuron:mean_soma_radius'}


def test_morph_stat_full_config():
    runner = CliRunner()
    filename = DATA / 'h5/v1/Neuron.h5'
    with tempfile.NamedTemporaryFile() as f:
        result = runner.invoke(cli, ['stats', str(filename), '--full-config', '--output', f.name])
        assert result.exit_code == 0
        df = pd.read_csv(f)
        assert not df.empty


def test_morph_stat_invalid_config():
    runner = CliRunner()
    with tempfile.NamedTemporaryFile('w') as config_f:
        yaml.dump({'neurite': 'invalid'}, config_f)
        result = runner.invoke(cli, ['stats', '--config', config_f.name])
        assert result.exit_code == 1
        assert isinstance(result.exception, ConfigError)


def test_morph_stat_stdout():
    runner = CliRunner()
    filename = DATA / 'swc' / 'simple.swc'
    result = runner.invoke(cli, ['stats', str(filename)])
    assert result.exit_code == 0


def test_morph_stat_as_population():
    runner = CliRunner()
    filename = DATA / 'swc' / 'simple.swc'
    result = runner.invoke(cli, ['stats', str(filename), '--as-population'])
    assert result.exit_code == 0


def test_morph_stat_json():
    runner = CliRunner()
    filename = DATA / 'swc' / 'simple.swc'
    with tempfile.NamedTemporaryFile(suffix='.json') as f:
        result = runner.invoke(cli, ['stats', str(filename), '--output', f.name])
        assert result.exit_code == 0
        content = json.load(f)
        assert content



def test_morph_check():
    runner = CliRunner()
    filename = DATA / 'swc' / 'simple.swc'
    with tempfile.NamedTemporaryFile() as f:
        result = runner.invoke(cli, ['check', str(filename), '--output', f.name])
        assert result.exit_code == 0
        content = json.load(f)
        assert content == {'files': {
            str(filename.absolute()): {'Has basal dendrite': True,
                                       'Has axon': True,
                                       'Has apical dendrite': False,
                                       'Has all nonzero segment lengths': True,
                                       'Has all nonzero section lengths': True,
                                       'Has all nonzero neurite radii': False,
                                       'Has nonzero soma radius': True,
                                       'ALL': False}},
            'STATUS': 'FAIL'}


def test_features():
    runner = CliRunner()
    result = runner.invoke(cli, ['features'])
    assert result.exit_code == 0
    assert len(result.stdout) > 0
