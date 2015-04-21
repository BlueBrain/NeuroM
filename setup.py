# pylint: disable=R0801
#!/usr/bin/env python
""" Distribution configuration for neurom
"""

from distutils.core import setup # pylint: disable=E0611,F0401

from pip.req import parse_requirements
from optparse import Option

from neurom.version import VERSION

OPTIONS = Option("--workaround")
OPTIONS.skip_requirements_regex = None
INSTALL_REQS = parse_requirements("./requirements.txt", options=OPTIONS)
REQS = [str(ir.req) for ir in INSTALL_REQS]

setup(name='neurom',
      version=VERSION,
      description='Neurom',
      install_requires=REQS,
      packages=['neurom', 'neurom.io', 'neurom.core'],
      )
