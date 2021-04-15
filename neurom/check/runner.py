#!/usr/bin/env python
# Copyright (c) 2015, Ecole Polytechnique Federale de Lausanne, Blue Brain Project
# All rights reserved.
#
# This file is part of NeuroM <https://github.com/BlueBrain/NeuroM>
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
#     1. Redistributions of source code must retain the above copyright
#        notice, this list of conditions and the following disclaimer.
#     2. Redistributions in binary form must reproduce the above copyright
#        notice, this list of conditions and the following disclaimer in the
#        documentation and/or other materials provided with the distribution.
#     3. Neither the name of the copyright holder nor the names of
#        its contributors may be used to endorse or promote products
#        derived from this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
# ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
# WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE FOR ANY
# DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
# (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
# LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
# ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
# (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

"""Runner for neuron morphology checks."""

import logging
from collections import OrderedDict
from importlib import import_module

from neurom import load_neuron
from neurom.check import check_wrapper
from neurom.exceptions import ConfigError
from neurom.io import utils

L = logging.getLogger(__name__)


class CheckRunner:
    """Class managing checks, config and output."""

    def __init__(self, config):
        """Initialize a CheckRunner object."""
        self._config = CheckRunner._sanitize_config(config)
        self._check_modules = dict((k, import_module('neurom.check.%s' % k))
                                   for k in config['checks'])

    def run(self, path):
        """Test a bunch of files and return a summary JSON report."""
        SEPARATOR = '=' * 40
        summary = {}
        res = True

        for _f in utils.get_files_by_path(path):
            L.info(SEPARATOR)
            status, summ = self._check_file(_f)
            res &= status
            if summ is not None:
                summary.update(summ)

        L.info(SEPARATOR)

        status = 'PASS' if res else 'FAIL'

        return {'files': summary, 'STATUS': status}

    def _do_check(self, obj, check_module, check_str):
        """Run a check function on obj."""
        opts = self._config['options']
        if check_str in opts:
            fargs = opts[check_str]
            if isinstance(fargs, list):
                out = check_wrapper(getattr(check_module, check_str))(obj, *fargs)
            else:
                out = check_wrapper(getattr(check_module, check_str))(obj, fargs)
        else:
            out = check_wrapper(getattr(check_module, check_str))(obj)

        try:
            if out.info:
                L.debug('%s: %d failing ids detected: %s',
                        out.title, len(out.info), out.info)
        except TypeError:  # pragma: no cover
            pass

        return out

    def _check_loop(self, obj, check_mod_str):
        """Run all the checks in a check_module."""
        check_module = self._check_modules[check_mod_str]
        checks = self._config['checks'][check_mod_str]
        result = True
        summary = OrderedDict()
        for check in checks:
            ok = self._do_check(obj, check_module, check)
            summary[ok.title] = ok.status
            result &= ok.status

        return result, summary

    def _check_file(self, f):
        """Run tests on a morphology file."""
        L.info('File: %s', f)

        full_result = True
        full_summary = OrderedDict()

        try:
            nrn = load_neuron(f)
            result, summary = self._check_loop(nrn, 'neuron_checks')
            full_result &= result
            full_summary.update(summary)
        except Exception as e:  # pylint: disable=W0703
            L.error('Check failed: %s', str(type(e)) + str(e.args))
            full_result = False

        full_summary['ALL'] = full_result

        for m, s in full_summary.items():
            self._log_msg(m, s)

        return full_result, {str(f): full_summary}

    def _log_msg(self, msg, ok):
        """Helper to log message to the right level."""
        if self._config['color']:
            CGREEN, CRED, CEND = '\033[92m', '\033[91m', '\033[0m'
        else:
            CGREEN = CRED = CEND = ''

        LOG_LEVELS = {False: logging.ERROR, True: logging.INFO}

        # pylint: disable=logging-not-lazy
        L.log(LOG_LEVELS[ok],
              '%35s %s' + CEND, msg, CGREEN + 'PASS' if ok else CRED + 'FAIL')

    @staticmethod
    def _sanitize_config(config):
        """Check that the config has the correct keys, add missing keys if necessary."""
        if 'checks' in config:
            checks = config['checks']
            if 'neuron_checks' not in checks:
                checks['neuron_checks'] = []
        else:
            raise ConfigError('Need to have "checks" in the config')

        if 'options' not in config:
            L.debug('Using default options')
            config['options'] = {}

        if 'color' not in config:
            config['color'] = False

        return config
