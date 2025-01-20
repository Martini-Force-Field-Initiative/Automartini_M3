"""
Basic version test for the auto_martini package.
"""
import pytest
import auto_martiniM3

try:
    from importlib import metadata
except ImportError:
    import importlib_metadata as metadata


def test_auto_martini_version():

    assert auto_martiniM3.__version__ == metadata.version("auto_martiniM3")
