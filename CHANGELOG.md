# Changelog

All notable changes to Verdict are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

## [0.1.0] — 2024-01-01

### Added

**Attack Implementations**
- **PAIR attack** (Chao et al., 2023): iterative LLM-based jailbreak refinement with configurable attacker model, temperature, and parallel streams
- **Crescendo attack** (Russinovich et al., 2024): multi-turn gradual escalation with LLM-planned setup and escalation question sequences, automatic refusal recovery
- **Template attacks**: 8 jailbreak templates — DAN, ResearcherAuthority, FictionalStory, HypotheticalWorld, EducationalPurpose, DocumentCompletion, Base64Cipher, AssistantSuffix
- **Prompt injection**: 8 direct templates (classic ignore, system override, XML injection, etc.) + 4 indirect templates (document, email, code review, search result injection)

**Semantic Judge**
- Binary YES/NO LLM judge with JSON structured output and confidence score
- 10-point rubric judge for ambiguous cases (score normalized to [0,1])
- Heuristic refusal pattern fallback (no LLM call required)
- Two-stage evaluation: binary first, rubric when score falls in ambiguous range

**Scenario Suite**
- 20 HarmBench-aligned scenarios across 7 harm categories
- Categories: CHEMICAL, CYBER, PHYSICAL, HARASSMENT, DISINFORMATION, ILLEGAL_WEAPONS, FINANCIAL_FRAUD, PRIVACY
- Severity tiers: CRITICAL, HIGH, MEDIUM, LOW
- Quick eval suite: 5 representative scenarios for CI gate (~3 min)

**Engine & Reporting**
- `RedTeamEngine` with async parallel evaluation, configurable concurrency
- `compare_models()` for baseline vs candidate safety regression testing
- ASR metrics: overall, per-category, per-attack-type, per-severity
- JSON report (machine-readable, CI integration)
- HTML report (dark-theme, Chart.js ASR bar charts, per-scenario table)
- Model comparison JSON with delta ASR and regression verdict

**API & CLI**
- FastAPI REST server: POST /evaluate, POST /compare, GET /scenarios, GET /attacks
- Rich CLI: evaluate, quick-eval, compare, list-scenarios, serve commands
- Auto-reload dev mode, configurable workers

**Infrastructure**
- Full async implementation with `asyncio`/`litellm`
- LiteLLM backend: supports OpenAI, Anthropic, Together AI, Groq, Ollama
- Exponential backoff retry on transient API errors
- Pytest test suite: 80+ CPU-only tests (mock judge + target)
- GitHub Actions CI: lint (ruff), type-check (mypy), tests (pytest)
