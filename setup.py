# -*- coding: utf-8 -*-
from pybind11.setup_helpers import Pybind11Extension, build_ext
from setuptools import setup

with open("README.md", "r") as fh:
    long_description = fh.read()

CPP_VERSION = 11


def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension("lcs", ["fuzzy_lightning/cpp/lcs.cpp"], cxx_std=CPP_VERSION),
        Pybind11Extension("edit_distance", ["fuzzy_lightning/cpp/edit_distance.cpp"], cxx_std=CPP_VERSION),
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmd_class": {"build_ext": build_ext},
            "zip_safe": False,
        }
    )


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
    'version': '0.1.6',
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
    'keywords': ['Fuzzy', 'String', 'Lookup', 'Search', 'Match', 'Similarity'],
}

build(setup_kwargs)

setup(**setup_kwargs)
