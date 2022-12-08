from pybind11.setup_helpers import Pybind11Extension, build_ext

CPP_VERSION = 11


def build(setup_kwargs):
    ext_modules = [
        Pybind11Extension("lcs", ["fast_fuzzy_matching/lcs/lcs.cpp"], cxx_std=CPP_VERSION),
    ]
    setup_kwargs.update(
        {
            "ext_modules": ext_modules,
            "cmd_class": {"build_ext": build_ext},
            "zip_safe": False,
        }
    )
