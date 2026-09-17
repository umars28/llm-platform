"""Fakes shared by more than one test module.

`BrokenRedis` lived in test_store.py and was imported from test_api.py as
`from tests.test_store import ...`, which only resolved because of where
sys.path happened to point when pytest was run from this directory. On a clean
checkout it raised ModuleNotFoundError: tests/ is not a package. A fake used by
two modules belongs in conftest, which pytest makes importable by construction.
"""

from __future__ import annotations


class BrokenRedis:
    """A client whose every call fails, to pin down the failure policy."""

    def register_script(self, _src):
        def script(keys=None, args=None):
            raise ConnectionError("redis is gone")
        return script

    def __getattr__(self, _name):
        def fail(*a, **k):
            raise ConnectionError("redis is gone")
        return fail
