import causal_testing

def test_package_import():
    # Just importing triggers __init__.py
    assert True

def test_version_attribute():
    assert hasattr(causal_testing, "__version__")
    assert isinstance(causal_testing.__version__, str)