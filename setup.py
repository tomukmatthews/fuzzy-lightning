# -*- coding: utf-8 -*-
from setuptools import setup

packages = ['fast_fuzzy_matching']

package_data = {'': ['*'], 'fast_fuzzy_matching': ['cpp/*']}

install_requires = [
    'pandas>=1.5.2,<2.0.0',
    'pybind11>=2.10.1,<3.0.0',
    'scikit-learn>=1.1.3,<2.0.0',
    'scipy>=1.9.3,<2.0.0',
    'setuptools>=65.6.3,<66.0.0',
    'sparse-dot-topn>=0.3.3,<0.4.0',
]

setup_kwargs = {
    'name': 'fast-fuzzy-matching',
    'version': '0.1.0',
    'description': 'Perform fast approximate string matching.',
    'long_description': '# fast-fuzzy-matching\nFast approximate string matching.\n',
    'author': 'Tom Matthews',
    'author_email': 'tomukmatthews@gmail.com',
    'maintainer': 'None',
    'maintainer_email': 'None',
    'url': 'None',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.10.7',
}
from build import *

build(setup_kwargs)

setup(**setup_kwargs)
