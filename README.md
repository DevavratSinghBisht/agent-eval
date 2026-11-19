# agent-eval

A research harness for tool-calling LLM agents. **Current scope:** BrowseComp (live web) and
BrowseComp-Plus (fixed ~100K-document corpus), Claude Haiku 5.5 as the first model, full-fidelity
logging, integrity validation, and failure analysis with strategic ablations. Runs on Windows in a
conda env — no Docker or WSL.

Design rationale (standards review of Anthropic / OpenAI / Google / Mercor agent guidance) is in the
approved plan; the short version: a **single tool-calling agent first**, with a planner–evaluator
loop added later as a drop-in strategy under identical tools, budgets and logging.

## Setup

```powershell
conda create -n agent-eval python=3.13 -y
conda activate agent-eval
pip install -e ".[dev]"
copy .env.example .env      # then fill ANTHROPIC_API_KEY and TAVILY_API_KEY (or EXA_API_KEY)
pytest                      # 20 offline tests (scripted model, mock web, mock corpus)
```

Optional, for the local-docs / oracle-docs experiments (downloads ~3 GB, builds a BM25 index):

```powershell
agent-eval prepare browsecomp_plus
```

## Running

```powershell
agent-eval run configs/experiments/pilot.yaml          # ~5 questions, $5 cap: always run this first
agent-eval trajectory runs/pilot <run_id> <task_id> --html t.html   # read the traces
agent-eval run configs/experiments/baseline.yaml       # 100 Qs x 3 repeats
agent-eval run configs/experiments/ablation.yaml       # tool-access conditions, paired questions
agent-eval run configs/experiments/delivery.yaml       # full vs truncated vs paginated vs summarized
agent-eval label runs/<campaign>                       # optional LLM failure taxonomy labels
agent-eval compare runs/<campaign> <config_hash_a> <config_hash_b>   # paired difference + McNemar
```

`run` is resumable: re-running the same YAML skips trials that already have a record
(`--retry-errored` re-runs errored/ungraded ones). Each run's results are finalized, validated and
reported automatically (`finalize`, `validate` and `report` are also separate commands).

## Experiments

**Tool conditions** (`condition`):

| id | tools | question it answers |
|---|---|---|
| `closed_book` | final_answer only | internal-knowledge floor / contamination |
| `search_snippets` | web_search (snippets only) | value of search ranking alone |
| `search_fetch` | web_search + web_fetch | standard browsing agent |
| `local_docs` | search_docs + read_doc (BrowseComp-Plus) | closed-corpus retrieval |
| `search_docs` | all four | combination effects |
| `oracle_docs` | read_doc on the gold evidence docs | reasoning ceiling with perfect retrieval |

**Delivery policies** (`delivery`) control what part of each tool result the model sees:
`full`, `truncate_2k|8k|32k`, `paginate_8k` (adds a `read_more` tool), `summarize` (LLM summary of long
results). Raw *and* delivered outputs are both logged for every tool call.

**Evidence funnel.** Every executed task is placed on `retrieved → delivered → attended → correct`.
A failure is attributed to the first broken link: `retrieval_miss`, `truncation_loss`,
`comprehension_miss`, `synthesis_error`, `format_or_grading`, `knowledge_miss` (closed book), or
`no_answer_<cap>`. Matching uses the normalized gold answer string. For BrowseComp-Plus it also
uses the gold and evidence docids.

## Output layout (`runs/<campaign>/`)

Modelled on the Mercor loss-analysis package in `example_logs/`:

| file | contents |
|---|---|
| `runs.csv` | one row per run: model, harness, condition, delivery, effort, budget, `config_hash`, `repeat_group_id`, `runner_version`, `price_catalog_version`, status, expected/executed/passed counts, token and cost totals. No stored score. |
| `tasks.csv` | full (run × task) grid **including `not_executed`**: status, passed, termination reason, tokens, cost, answer, confidence, grader output, funnel flags and category, failure signals |
| `model_calls.csv` | one row per model call: `call_index`, `harness_phase`, uncached / cached / cache-write / completion tokens, stop reason, latency, `exception_type`, cost |
| `tool_calls.csv` | one row per tool call: raw vs delivered tokens, truncated, pages, answer-in-raw / answer-in-delivered, gold-doc hit, fees |
| `trajectories/archive.jsonl.zst` | one record per task. Each record has one step per model call (same order, phase and usage as `model_calls.csv`) plus a closing step. Steps hold reasoning, text, raw response, and each tool call's input with its raw output, delivered output and delivery metadata. |
| `trajectories/trajectory_index.csv`, `selected/` | index with byte offsets; exemplars across failure categories |
| `traces/otel_spans.jsonl.zst` | the same data as OpenTelemetry GenAI spans (`invoke_agent` → `chat` → `execute_tool`) |
| `_work/<run>/raw_api/*.jsonl.zst` | provider responses plus request deltas for replay |
| `validation.json`, `report.md`, `manifest.json` | integrity findings, analysis report, campaign metadata |

**Conventions**
- `prompt_tokens` excludes cached tokens.
- Task `cost_usd` = model calls + tool fees (search, summarizer). Grader cost is reported separately.
- Costs always come from the versioned `configs/price_catalog.yaml`. An unpriced model is a hard error, never a fallback rate.
- Run score = `tasks_passed / tasks_expected`. Not-executed and errored tasks count as failures; executed-only accuracy is shown next to it.
- Repeats are runs with the same `config_hash`. Configs are compared with task-level bootstrap CIs, pass@k / pass^k, and paired tests on shared tasks.

**Validator** (`agent-eval validate`)
- Rollups reconcile exactly: calls → tasks → runs, tokens and cost.
- Trajectory steps = calls + 1.
- No missing runner or catalog version.
- Every call reprices to its logged cost from the catalog.
- Flags unverifiable runs (passes with no call log) and infrastructure failures (mostly failed calls or errored tasks). Both are excluded from performance tables.
- Flags partial runs, mixed runner versions, cached-share anomalies between repeats, and grader parse failures.

## Notes and assumptions to verify
- **simple-evals bug fixed:** its grader compares `match.group(0)` (`"correct: yes"`) to `"yes"`, so it never scores a correct answer. We compare the captured group.
- **Price catalog:** Haiku 5.5 prices come from Anthropic's model table. Two things are assumptions: the cache multipliers (read 0.1×, write 1.25×), and that the >100K-prompt tier applies per call. Check them against the pricing page and bump `version` if they change.
- **Search backends:** Tavily and Exa are implemented against their public REST APIs but haven't been exercised with real keys yet. The pilot will confirm them.
- **Answer-leak blocklist:** search results and fetches skip URLs that mirror benchmark data (`tools/web.py`, extendable via `AGENT_EVAL_BLOCKLIST`).
- **Commit before campaigns:** `runner_version` includes the git SHA, and shows `nogit` / `-dirty` otherwise.

## Roadmap
- **Phase 4:** OpenAI, Gemini and OpenAI-compatible (vLLM / Ollama) providers behind `providers/base.py`, plus a provider conformance suite.
- **Phase 5:** `planner_evaluator` harness (planner → executor → evaluator loop), compared to `direct_tool` in a crossed, matched design.
- **Later:** APEX-Agents (Corporate Lawyer) and APEX-Accounting through Mercor's Harbor runtime (needs Docker).
