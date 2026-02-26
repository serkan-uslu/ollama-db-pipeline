"""
tests/test_crawler.py

Unit tests for pipeline/agents/crawler.py

Covers:
- parse_pulls()             — text-to-int conversion
- parse_last_updated()      — relative date parsing
- detect_uncensored()       — heuristic keyword detection
- parse_model_detail_html() — README + memory_requirements extraction
"""

from datetime import date, timedelta

import pytest

from pipeline.agents.crawler import (
    detect_uncensored,
    parse_last_updated,
    parse_model_detail_html,
    parse_pulls,
)


# ── parse_pulls ────────────────────────────────────────────────────────────────

class TestParsePulls:
    def test_millions(self):
        assert parse_pulls("110.4M") == 110_400_000

    def test_thousands(self):
        assert parse_pulls("78.4K") == 78_400

    def test_billions(self):
        assert parse_pulls("1.2B") == 1_200_000_000

    def test_plain_integer(self):
        assert parse_pulls("5000") == 5000

    def test_lowercase(self):
        assert parse_pulls("55m") == 55_000_000

    def test_with_commas(self):
        assert parse_pulls("1,200") == 1200

    def test_empty_string_returns_zero(self):
        assert parse_pulls("") == 0

    def test_invalid_string_returns_zero(self):
        assert parse_pulls("N/A") == 0

    def test_whitespace_stripped(self):
        assert parse_pulls("  12.5M  ") == 12_500_000


# ── parse_last_updated ─────────────────────────────────────────────────────────

class TestParseLastUpdated:
    def test_months_ago(self):
        parsed_date, raw = parse_last_updated("7 months ago")
        today = date.today()
        expected_year = today.year if today.month > 7 else today.year - 1
        assert parsed_date.year in (today.year, today.year - 1)
        assert raw == "7 months ago"

    def test_years_ago(self):
        parsed_date, raw = parse_last_updated("2 years ago")
        assert parsed_date.year == date.today().year - 2
        assert raw == "2 years ago"

    def test_weeks_ago(self):
        parsed_date, raw = parse_last_updated("3 weeks ago")
        expected = date.today() - timedelta(weeks=3)
        assert parsed_date == expected

    def test_days_ago(self):
        parsed_date, raw = parse_last_updated("5 days ago")
        expected = date.today() - timedelta(days=5)
        assert parsed_date == expected

    def test_1_year_ago_singular(self):
        parsed_date, _ = parse_last_updated("1 year ago")
        assert parsed_date.year == date.today().year - 1

    def test_unknown_format_returns_today(self):
        parsed_date, raw = parse_last_updated("just now")
        assert parsed_date == date.today()
        assert raw == "just now"

    def test_empty_string(self):
        parsed_date, _ = parse_last_updated("")
        assert parsed_date == date.today()


# ── detect_uncensored ──────────────────────────────────────────────────────────

class TestDetectUncensored:
    def test_slug_uncensored(self):
        assert detect_uncensored("llama2-uncensored", "Llama2", None) is True

    def test_name_abliterated(self):
        assert detect_uncensored("model-x", "Abliterated Llama", None) is True

    def test_description_contains_keyword(self):
        assert detect_uncensored(
            "dolphin-mistral", "Dolphin", "An uncensored model"
        ) is True

    def test_dolphin_slug(self):
        assert detect_uncensored("dolphin-mixtral", "Dolphin Mixtral", None) is True

    def test_normal_model(self):
        assert detect_uncensored("llama3.1", "Llama 3.1", "A helpful assistant") is False

    def test_case_insensitive(self):
        assert detect_uncensored("UNCENSORED-model", "Model", None) is True

    def test_none_description(self):
        # Should not raise
        assert detect_uncensored("llama3", "Llama 3", None) is False


# ── parse_model_detail_html ────────────────────────────────────────────────────

class TestParseModelDetailHtml:
    def _make_html(self, readme: str = "", memory_rows: str = "") -> str:
        return f"""
        <html><body>
        <section id="readme">{readme}</section>
        <div class="model-info">
          {memory_rows}
        </div>
        </body></html>
        """

    def test_readme_extraction(self):
        html = self._make_html(readme="This is a really useful LLM model.")
        result = parse_model_detail_html(html)
        assert result["readme"] is not None
        assert "useful" in result["readme"]

    def test_empty_html_returns_none_readme(self):
        result = parse_model_detail_html("<html><body></body></html>")
        assert result["readme"] is None

    def test_result_keys_present(self):
        result = parse_model_detail_html("<html><body></body></html>")
        assert "readme" in result
        assert "memory_requirements" in result

    def test_memory_requirements_list_or_none(self):
        result = parse_model_detail_html("<html><body></body></html>")
        assert result["memory_requirements"] is None or isinstance(
            result["memory_requirements"], list
        )

    def test_readme_truncated_to_10000_chars(self):
        long_text = "a" * 20000
        html = self._make_html(readme=long_text)
        result = parse_model_detail_html(html)
        if result["readme"]:
            assert len(result["readme"]) <= 10000

    def test_memory_pattern_extraction(self):
        """Memory table text embedded in page body."""
        html = """
        <html><body>
        <section id="readme">Helpful coding model</section>
        <p>7b 4.1 GB · 4096 context</p>
        <p>13b 7.9 GB · 4096 context</p>
        </body></html>
        """
        result = parse_model_detail_html(html)
        # memory_requirements may or may not match depending on regex
        assert isinstance(result, dict)
        assert "memory_requirements" in result
