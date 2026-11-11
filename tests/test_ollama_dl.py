import pytest

from ollama_dl import format_size, get_short_hash


def test_get_short_hash_returns_12_char_prefix():
    layer = {"digest": "sha256:abcdef1234567890"}
    assert get_short_hash(layer) == "abcdef123456"


def test_get_short_hash_rejects_non_sha256_digest():
    with pytest.raises(ValueError):
        get_short_hash({"digest": "md5:abcdef1234567890"})


def test_get_short_hash_missing_digest_raises_value_error():
    with pytest.raises(ValueError):
        get_short_hash({})


def test_format_size_bytes():
    assert format_size(512) == "512 B"


def test_format_size_kilobytes():
    assert format_size(2048) == "2 KB"


def test_format_size_megabytes():
    assert format_size(3 * 1024 * 1024) == "3 MB"
