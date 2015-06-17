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

ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

neurom_test_venv:
	virtualenv --system-site-packages neurom_test_venv
	neurom_test_venv/bin/pip install --upgrade --force-reinstall pep8
	neurom_test_venv/bin/pip install --upgrade --force-reinstall pylint
	neurom_test_venv/bin/pip install --upgrade --force-reinstall nose
	neurom_test_venv/bin/pip install -e .

run_pep8: neurom_test_venv
	neurom_test_venv/bin/pep8 --config=pep8rc `find neurom examples -name "*.py" -not -path "./*venv*/*" -not -path "*/*test*"` > pep8.txt

run_pylint: neurom_test_venv
	neurom_test_venv/bin/pylint --rcfile=pylintrc `find neurom examples -name "*.py" -not -path "./*venv*/*" -not -path "*/*test*"` > pylint.txt

run_tests: neurom_test_venv
	neurom_test_venv/bin/nosetests -v --with-coverage --cover-package neurom

run_tests_xunit: neurom_test_venv
	@mkdir -p $(ROOT_DIR)/test-reports
	neurom_test_venv/bin/nosetests neurom --with-coverage --cover-inclusive --cover-package=neurom  --with-xunit --xunit-file=test-reports/nosetests_neurom.xml

lint: run_pep8 run_pylint

test: run_tests

doc:  neurom_test_venv
	neurom_test_venv/bin/pip install --upgrade sphinx sphinxcontrib-napoleon
	make SPHINXBUILD=$(ROOT_DIR)/neurom_test_venv/bin/sphinx-build -C doc html

clean_test_venv:
	@rm -rf neurom_test_venv
	@rm -rf $(ROOT_DIR)/test-reports

clean_doc:
	@test -x $(ROOT_DIR)/neurom_test_venv/bin/sphinx-build && make SPHINXBUILD=$(ROOT_DIR)/neurom_test_venv/bin/sphinx-build  -C doc clean || true

clean: clean_doc clean_test_venv
	@rm -f pep8.txt
	@rm -f pylint.txt

.PHONY: run_pep8 test clean_test_venv clean doc
