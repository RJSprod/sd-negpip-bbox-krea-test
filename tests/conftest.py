"""Import the Extension's package the way Forge does: by path, under its own name."""

import importlib
import importlib.util
import os
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBRARY = os.path.join(ROOT, "lib_negpip_regional")
PACKAGE = "lib_negpip_regional_tests"


def _load():
    if PACKAGE in sys.modules:
        return sys.modules[PACKAGE]

    spec = importlib.util.spec_from_file_location(
        PACKAGE, os.path.join(LIBRARY, "__init__.py"),
        submodule_search_locations=[LIBRARY])
    module = importlib.util.module_from_spec(spec)
    sys.modules[PACKAGE] = module
    spec.loader.exec_module(module)
    return module


_load()


@pytest.fixture(scope="session")
def regions():
    return importlib.import_module(f"{PACKAGE}.regions")


@pytest.fixture(scope="session")
def regional():
    return importlib.import_module(f"{PACKAGE}.regional")
