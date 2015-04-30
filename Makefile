ROOT_DIR := $(shell dirname $(realpath $(lastword $(MAKEFILE_LIST))))

neurom_test_venv:
	virtualenv --system-site-packages neurom_test_venv
	neurom_test_venv/bin/pip install --upgrade --force-reinstall pep8
	neurom_test_venv/bin/pip install --upgrade --force-reinstall pylint
	neurom_test_venv/bin/pip install --upgrade --force-reinstall nose
	neurom_test_venv/bin/pip install -e .

run_pep8: neurom_test_venv
	neurom_test_venv/bin/pep8 --config=pep8rc `find neurom -name "*.py" -not -path "./*venv*/*" -not -path "*/*test*"` > pep8.txt

run_pylint: neurom_test_venv
	neurom_test_venv/bin/pylint --rcfile=pylintrc `find neurom -name "*.py" -not -path "./*venv*/*" -not -path "*/*test*"` > pylint.txt

run_tests: neurom_test_venv
	neurom_test_venv/bin/nosetests -v --with-coverage --cover-package neurom

lint: run_pep8 run_pylint

test: run_tests

doc: 
	neurom_test_venv/bin/pip install sphinx sphinxcontrib-napoleon
	make SPHINXBUILD=$(ROOT_DIR)/neurom_test_venv/bin/sphinx-build -C doc html

clean_test_venv:
	/bin/rm -rf neurom_test_venv

clean_doc:
	make SPHINXBUILD=$(ROOT_DIR)/neurom_test_venv/bin/sphinx-build  -C doc clean

clean: clean_test_venv clean_doc
	/bin/rm -f pep8.txt

.PHONY: run_pep8 test clean_test_venv clean doc
