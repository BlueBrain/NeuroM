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

#see if we're in a virtualenv, and use that, otherwise use the default
ifdef VIRTUAL_ENV
   VENV=$(VIRTUAL_ENV)
else
   VENV:=venv
endif
VENV_BIN:=$(VENV)/bin

# simulate running in headless mode
unexport DISPLAY

# Test coverage pass threshold (percent)
MIN_COV?=100
PIP_VERSION:=8.1.2
VENV_INSTALLED=.installed

FIND_LINT_PY=`find neurom examples apps -name "*.py" -not -path "*/*test*"`
FIND_LINT_APP=`find apps -type f -not -path "*/*\.*" -not -path "*/\.*" -not -path "*/.*~"`
LINT_PYFILES := $(shell find $(FIND_LINT_PY)) $(shell find $(FIND_LINT_APP))

$(VENV):
	virtualenv --system-site-packages $(VENV)

$(VENV_INSTALLED): $(VENV)
	$(VENV_BIN)/pip install --upgrade pip==$(PIP_VERSION)
	$(VENV_BIN)/pip install --ignore-installed -r requirements_dev.txt
	$(VENV_BIN)/pip install -e .
	touch $@

run_pep8: $(VENV_INSTALLED)
	$(VENV_BIN)/pep8 --config=pep8rc $(LINT_PYFILES) > pep8.txt

run_pylint: $(VENV_INSTALLED)
	$(VENV_BIN)/pylint --rcfile=pylintrc --extension-pkg-whitelist=numpy $(LINT_PYFILES) > pylint.txt

run_tests: $(VENV_INSTALLED)
	$(VENV_BIN)/nosetests -v --with-coverage --cover-min-percentage=$(MIN_COV) --cover-erase --cover-package neurom

run_tests_xunit: $(VENV_INSTALLED)
	@mkdir -p $(ROOT_DIR)/test-reports
	$(VENV_BIN)/nosetests neurom --with-coverage --cover-min-percentage=$(MIN_COV) --cover-inclusive --cover-erase --cover-package=neurom  --with-xunit --xunit-file=test-reports/nosetests_neurom.xml

lint: run_pep8 run_pylint

test: lint run_tests

doc: $(VENV_INSTALLED)
	make SPHINXBUILD=$(ROOT_DIR)/$(VENV_BIN)/sphinx-build -C doc html

doc_pdf: $(VENV_INSTALLED)
	make SPHINXBUILD=$(ROOT_DIR)/$(VENV_BIN)/sphinx-build -C doc latexpdf

clean_test_venv:
	@rm -rf $(VENV_INSTALLED)
	@rm -rf $(ROOT_DIR)/test-reports

clean_doc:
	@test -x $(ROOT_DIR)/$(VENV_BIN)/sphinx-build && make SPHINXBUILD=$(ROOT_DIR)/$(VENV_BIN)/sphinx-build  -C doc clean || true
	@rm -rf $(ROOT_DIR)/doc/source/_neurom_build

clean: clean_doc clean_test_venv
	@rm -f pep8.txt
	@rm -f pylint.txt
	@rm -rf neurom.egg-info
	@rm -f .coverage
	@rm -rf test-reports
	@rm -rf dist
	@rm -f $(VENV_INSTALLED)

.PHONY: run_pep8 test clean_test_venv clean doc
