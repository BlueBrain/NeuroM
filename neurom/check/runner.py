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

'''runner for neuron morphology checks'''

from collections import OrderedDict
from importlib import import_module
import os
import logging
from neurom.io.utils import get_morph_files
from neurom.io import load_data
from neurom.fst import _core as fst_core
from neurom.check import check_wrapper

L = logging.getLogger(__name__)


class CheckRunner(object):
    '''Class managing checks, config and output'''
    def __init__(self, config):
        self._config = config
        if 'color' not in self._config:
            self._config['color'] = False
        self.summary = OrderedDict()
        self._check_modules = dict((k, import_module('neurom.check.%s' % k))
                                   for k in config['checks'])

    def run(self, file_path):
        '''Test a bunch of files and return a summary JSON report'''

        def _get_files():
            '''Get a file or set of files from a file path'''
            if os.path.isfile(file_path):
                return [file_path]
            elif os.path.isdir(file_path):
                L.info('Checking files in directory %s', file_path)
                return get_morph_files(file_path)
            else:
                msg = 'Invalid data path %s' % file_path
                L.error(msg)
                raise IOError(msg)

        SEPARATOR = '=' * 40
        summary = {}
        res = True

        for _f in _get_files():
            L.info(SEPARATOR)
            status, summ = self._check_file(_f)
            res &= status
            if summ is not None:
                summary.update(summ)

        L.info(SEPARATOR)

        status = 'PASS' if res else 'FAIL'

        return {'files': summary, 'STATUS': status}

    def _do_check(self, obj, check_module, check_str):
        '''Run a check function on obj'''
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
            if len(out.info) > 0:
                L.debug('%s: %d failing ids detected: %s',
                        out.title, len(out.info), out.info)
        except TypeError:
            pass

        return out

    def _check_loop(self, obj, check_mod_str):
        '''Run all the checks in a check_module'''
        check_module = self._check_modules[check_mod_str]
        checks = self._config['checks'][check_mod_str]
        result = True
        for check in checks:
            ok = self._do_check(obj, check_module, check)
            self.summary[ok.title] = ok.status
            result &= ok.status

        return result

    def _check_file(self, f):
        '''Run tests on a morphology file'''

        L.info('File: %s', f)

        result = True

        try:
            data = load_data(f)
        except StandardError as e:
            L.error('Failed to load data... skipping tests for this file')
            L.error(e.message)
            return False, {f: OrderedDict([('ALL', False)])}

        try:
            result &= self._check_loop(data, 'structural_checks')
            nrn = fst_core.FstNeuron(data)
            result &= self._check_loop(nrn, 'neuron_checks')
        except StandardError as e:
            L.error('Check failed: %s: %s', type(e), e.message)
            result = False

        self.summary['ALL'] = result

        for m, s in self.summary.iteritems():
            self._log_msg(m, s)

        return result, {f: self.summary}

    def _log_msg(self, msg, ok):
        '''Helper to log message to the right level'''
        if self._config['color']:
            CGREEN, CRED, CEND = '\033[92m', '\033[91m', '\033[0m'
        else:
            CGREEN = CRED = CEND = ''

        LOG_LEVELS = {False: logging.ERROR, True: logging.INFO}

        L.log(LOG_LEVELS[ok],
              '%35s %s' + CEND, msg, CGREEN + 'PASS' if ok else CRED + 'FAIL')
