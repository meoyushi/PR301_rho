def test_package_imports():
    import rho
    from rho.config import settings

    assert settings.app_name == "rho"
