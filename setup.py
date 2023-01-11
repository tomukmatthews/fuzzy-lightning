# -*- coding: utf-8 -*-
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

packages = ['fuzzy_lightning']

package_data = {'': ['*'], 'fuzzy_lightning': ['cpp/*']}

install_requires = [
    'pybind11>=2.6.0',
    'scikit-learn>=1.0.2',
    'scipy>=1.7.0',
    'sparse-dot-topn>=0.3.1',
]

setup_kwargs = {
    'name': 'fuzzy-lightning',
    'version': '0.1.5',
    'description': 'Perform fast fuzzy string lookups.',
    'long_description': long_description,
    'long_description_content_type': 'text/markdown',
    'author': 'Tom Matthews',
    'author_email': 'tomukmatthews@gmail.com',
    'maintainer': 'Tom Matthews',
    'maintainer_email': 'tomukmatthews@gmail.com',
    'url': 'https://github.com/tomukmatthews/fuzzy-lightning',
    'packages': packages,
    'package_data': package_data,
    'install_requires': install_requires,
    'python_requires': '>=3.8.1,<4.0.0',
}
from build import build

build(setup_kwargs)

setup(**setup_kwargs)
