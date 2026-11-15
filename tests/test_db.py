import pytest
import re

import db


def test_database_password_is_required(monkeypatch):
    monkeypatch.delenv("POSTGRES_PASSWORD", raising=False)
    monkeypatch.delenv("DB_PASSWORD", raising=False)

    with pytest.raises(RuntimeError, match="POSTGRES_PASSWORD"):
        db._required_setting("POSTGRES_PASSWORD", "DB_PASSWORD")


def test_legacy_database_env_names_remain_supported(monkeypatch):
    monkeypatch.delenv("POSTGRES_USER", raising=False)
    monkeypatch.setenv("DB_USER", "local-user")

    assert db._required_setting("POSTGRES_USER", "DB_USER") == "local-user"


def test_database_pool_uses_configured_sslmode(monkeypatch):
    settings = {}

    class FakePool:
        def __init__(self, **kwargs):
            settings.update(kwargs)

    monkeypatch.setattr(db, "ThreadedConnectionPool", FakePool)
    monkeypatch.setenv("POSTGRES_DB", "test-db")
    monkeypatch.setenv("POSTGRES_USER", "test-user")
    monkeypatch.setenv("POSTGRES_PASSWORD", "test-password")
    monkeypatch.setenv("DB_SSLMODE", "require")
    monkeypatch.setattr(db, "_POOL", None)

    db._get_pool()

    assert settings["sslmode"] == "require"


def test_clause_pattern_matches_exact_id_not_longer_prefix():
    pattern = db._clause_patterns(["A.5.15"])[0]

    assert re.search(pattern, "Requirement A.5.15 applies", re.IGNORECASE)
    assert not re.search(pattern, "Requirement A.5.150 applies", re.IGNORECASE)


def test_schema_uses_immutable_full_text_configuration():
    assert "to_tsvector('english'::regconfig, text)" in db.SCHEMA_SQL
    assert "section_ids TEXT[]" in db.SCHEMA_SQL


def test_keyword_search_sql_placeholders_match_parameters():
    import inspect

    source = inspect.getsource(db._keyword_search_cursor)
    sql = source.split('"""', 1)[1].split('"""', 1)[0]
    assert sql.count("%s") == 12