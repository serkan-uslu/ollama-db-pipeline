## 📝 Pull Request Description

Briefly describe what this PR changes in the pipeline.

## 🎯 Type of Change

- [ ] Bug fix — fixes a broken pipeline step
- [ ] New feature — adds new functionality to an agent
- [ ] Enrichment improvement — better prompts, schemas or LLM logic
- [ ] Crawler improvement — better scraping / parsing logic
- [ ] Validation rule change
- [ ] Breaking change — changes DB schema or `models.json` output shape
- [ ] Documentation update
- [ ] Refactoring (no functional changes)
- [ ] CI / GitHub Actions change
- [ ] Other (please describe)

## ✅ Checklist

- [ ] I have run `pytest tests/ -v` and all tests pass
- [ ] I have run the affected agent in isolation to verify locally
- [ ] If I changed DB schema: I updated `pipeline/core/models.py` **and** documented the migration
- [ ] If I changed `EnrichmentOutput` schemas: `ENRICH_VERSION` has been bumped in `.env.example`
- [ ] If I changed `models.json` output shape: I notified the [ollama-explorer](https://github.com/serkan-uslu/ollama-explorer) team (the frontend consumes this file)
- [ ] My changes generate no new warnings or errors in the pipeline log
- [ ] I have added or updated tests for the changed code
- [ ] I have updated the relevant documentation (README / FEATURES.md / INSTRUCTIONS.md)

## 🔗 Related Issues

Fixes #
Related to #

## 🤖 Agent(s) Affected

- [ ] Crawler (`crawler.py`)
- [ ] Enricher (`enricher.py`)
- [ ] Validator (`validator.py`)
- [ ] Exporter (`exporter.py`)
- [ ] PR Creator (`pr_creator.py`)
- [ ] Flow orchestration (`flow.py`)
- [ ] Core / DB / Settings
- [ ] GitHub Actions workflow

## 🧪 Testing

Describe how you tested the change:

```bash
# Example: how you ran the relevant step
poetry run python main.py --skip-crawl --dry-run --model llama3.2
```

## 📊 Output Sample (if applicable)

If this affects `models.json` output, paste a before/after snippet of a single model entry.

## 💬 Additional Notes

Any additional context, tradeoffs, or follow-up work.
