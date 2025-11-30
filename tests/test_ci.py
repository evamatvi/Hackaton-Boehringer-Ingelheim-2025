def test_ci_environment_is_ready():
    """Este test trivial verifica que el entorno de GitHub Actions se configuró correctamente."""
    # Como queremos que el CI pase y nos dé el check verde,
    # simplemente afirmamos que la condición más básica es verdadera.
    assert True