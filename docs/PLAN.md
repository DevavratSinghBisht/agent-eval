# Plan — Tool-calling agent eval harness (BrowseComp first; APEX later)

## Context
`D:\Projects\Claude Projects\agent-eval` is empty. Goal: a research-grade harness that (1) runs a tool-calling agent on **Claude Haiku 5.5** (`claude-haiku-5-5`) against **BrowseComp**, (2) logs every step with full fidelity, (3) supports deep failure analysis **and strategic ablations** (closed-book vs search vs local docs vs combinations; full vs truncated tool-result consumption), (4) becomes provider-agnostic (OpenAI, Gemini, vLLM/Ollama), and (5) adds a planner–evaluator loop compared head-to-head with the plain tool-calling agent.

**Scope now:** BrowseComp (+ BrowseComp-Plus fixed corpus for the local-docs experiments). **No Docker, no WSL** — everything runs in a **conda env** on Windows (Python 3.13 via existing miniconda). APEX-Agents (Corporate Lawyer) and APEX-Accounting are deferred; the architecture keeps a `benchmarks/` + `tools/mcp_bridge` slot for them (they require Mercor's Harbor/Docker runtime, to be revisited later).

---

## 1. Standards review → design decision

| Source | What binds our design |
|---|---|
| **Anthropic** (Building effective agents; Writing tools for agents; Multi-agent research system; Context engineering; Demystifying evals) | Simplest thing first: single agent + good tools; evaluator-optimizer only when criteria are clear and evals show gains. Few high-signal tools, rich descriptions, actionable errors, truncation/pagination with sane defaults. Context editing/compaction. Eval vocabulary task/trial/transcript/outcome; isolated trials; **pass@k & pass^k**; read transcripts. On BrowseComp, token spend explains ~80% of score variance → always log tokens/cost. |
| **OpenAI** (Practical guide to agents; Agents SDK; trace grading; simple-evals) | Single agent first. Run loop exits on final output / error / max turns. Traces → typed spans (agent, generation, function, guardrail, custom). Grade outcomes + whole traces. BrowseComp reference impl = `openai/simple-evals/browsecomp_eval.py`. |
| **Google** (ADK, Agents Companion, Vertex agent eval) | Critic/refiner = `LoopAgent(max_iterations)` + explicit exit; state passed by named keys. Evaluate outside-in (outcome first, then trajectory), always-on efficiency counters (tool calls, LLM calls, tokens, duration). |
| **Mercor** (APEX-Agents paper/harness) | Minimal ReAct loop with **explicit `final_answer` tool**, tool-output truncation (200 lines / 32K chars), hard step/time caps, wrap-up nudge near budget, parse-error retry. Failure modes to detect: scattergunning (hedged multi-answers), doom loops, cap timeouts, ignoring available context. |
| **Cross-provider** | OpenTelemetry **GenAI semantic conventions** (`invoke_agent` → `chat` → `execute_tool`, `gen_ai.*` attrs); vLLM per-family tool parsers; thinking-block / thought-signature round-trips must be tested per provider. |

**Decision:** Build the **single tool-calling agent first** (universal recommendation), with the **planner–evaluator** as a drop-in *agent strategy* run under the identical tools, budgets and logging (Phase 5), so the comparison is apples-to-apples.

---

## 2. Environment & prerequisites (no Docker/WSL)
- Machine: Core Ultra 7 258V (8 cores), 31.6 GB RAM, D: 527 GB free — ample (BrowseComp-Plus corpus ~100K docs + BM25 index fits in a few GB RAM).
- `conda create -n agent-eval python=3.13` → `pip install -e .` from `pyproject.toml` (deps: `anthropic`, `httpx`, `pydantic`, `pyyaml`, `python-dotenv`, `rich`, `pandas`, `pyarrow`, `datasets`, `bm25s` (pure-Python BM25, no Java), `trafilatura` (HTML→text), `tiktoken`-free token estimates via Anthropic `count_tokens`, `pytest`). Export `environment.yml` for reproducibility.
- Keys in `.env` (gitignored): `ANTHROPIC_API_KEY` (or `ant auth login`), one search API key (pluggable backend: **Tavily / Exa / Brave / Serper**; implement Tavily + Exa first).
- BrowseComp grader model: configurable; default `claude-sonnet-5-5` (different model from the agent to limit self-preference; grading is short so cost is small).

## 3. Budget (estimates → replaced by measured $/trial after a 5-question pilot)
Haiku 5.5: $0.10 in / $0.50 out per MTok (prompts ≤100K), $0.50 / $2.50 above 100K — so context management directly controls cost.

| Run | Est. $/trial (model + search + grader) | First pass | Est. total |
|---|---|---|---|
| Pilot | — | 5 Qs × 1 | < $5 |
| BrowseComp baseline (C2, full delivery) | $0.25–0.45 | 100 Qs × 3 trials | ~$75–135 |
| Ablation matrix (C0–C5, BrowseComp-Plus where applicable) | $0.02–0.45 | 50 Qs × 1 × 6 conditions | ~$40–90 |
| Delivery-policy sweep (5 policies on C2/C3) | $0.2–0.45 | 50 Qs × 1 × 5 | ~$50–110 |

Suggested envelope: **~$250–350** for baseline + first ablation pass. Runner enforces a hard `max_cost_usd` per experiment.

---

## 4. Repo layout
```
agent-eval/
  pyproject.toml  environment.yml  .env.example  README.md
  configs/ models/haiku-5-5.yaml  benchmarks/browsecomp.yaml  experiments/{baseline,ablation,delivery}.yaml
  src/agent_eval/
    core/types.py          # neutral Message, ContentBlock, ToolCall, ToolResult, Usage, StopReason
    providers/base.py      # ModelClient protocol: complete(messages, tools, cfg) -> AssistantTurn
    providers/anthropic.py # Phase 1
    providers/{openai,gemini,openai_compatible}.py   # Phase 4
    tools/base.py          # ToolSpec + ToolResult envelope {raw_content, delivered_content, delivery_meta}
    tools/delivery.py      # DeliveryPolicy: full | truncate(N, head|head+tail) | paginate(N)+read_more | summarize
    tools/web_search.py    # pluggable backends (Tavily, Exa) + answer-leak domain blocklist
    tools/web_fetch.py     # httpx + trafilatura; returns full cleaned text (delivery policy applied after)
    tools/local_docs.py    # search_docs / read_doc over BrowseComp-Plus corpus (bm25s index cached on D:)
    tools/final_answer.py  # explicit termination tool: explanation, exact_answer, confidence
    agent/react.py         # Strategy 1: tool loop, budgets (steps/tokens/time/$), wrap-up nudge, parse-error retry
    agent/context.py       # tool-result clearing / compaction policies (every action logged)
    agent/planner_eval.py  # Strategy 2 (Phase 5)
    benchmarks/browsecomp.py       # simple-evals port: CSV, canary XOR decrypt, QUERY/GRADER templates, regex parse, calibration
    benchmarks/browsecomp_plus.py  # Tevatron/browsecomp-plus (+ -corpus): decrypt, gold evidence docids (qrels)
    tracing/               # OTel-GenAI-shaped span JSONL writer (+ optional OTLP exporter later)
    runner/                # experiment matrix, trials, asyncio concurrency, resume, cost guardrails
    analysis/              # metrics, signals, evidence attribution, LLM failure labeller, reports, HTML viewer
  tests/
  runs/<experiment>/<condition>/<task_id>/<trial>/  trace.jsonl transcript.json raw_api/ outcome.json grade.json
```

---

## 5. Phases

### Phase 1 — Claude tool-calling agent on BrowseComp
- `AnthropicClient`: Haiku 5.5, `thinking: {type: "adaptive"}`, explicit `output_config.effort` (config variable; Haiku default is medium), streaming + `get_final_message()`, `tool_choice: auto`, `strict: true` tool schemas, prompt caching on frozen tools+system prefix (verify `cache_read_input_tokens`), handle `refusal` / `max_tokens` / `pause_turn`, parallel tool results returned in one user message, append-only history.
- `ReactAgent`: tools from condition config; must end via `final_answer` (explanation, exact_answer, confidence) mapped onto the official `Explanation / Exact Answer / Confidence` format; caps (default 50 steps, 1.5M tokens, 20 min, $ cap), wrap-up nudge at ~85% budget.
- BrowseComp: faithful port of simple-evals (download CSV, decrypt with row canary, QUERY_TEMPLATE, GRADER_TEMPLATE copied verbatim from the raw file, `correct: (yes|no)` parse, accuracy + calibration error). Fix the known bug (report shows encrypted answer). Search tool blocks domains known to republish BrowseComp answers.
- Concurrency: asyncio, 8–16 trials in parallel with rate-limit backoff.

### Phase 2 — Logging & tracing (built into Phase 1, hardened here)
Modelled on `example_logs/original_docs` (Mercor loss-analysis package: `runs.csv` → `tasks.csv` → `model_calls.csv` + trajectories). Same three-table + trajectory layout, extended for our experiments, and designed to avoid the data faults that package's analysis notebook had to catch.

**Campaign output layout** (`runs/<campaign_id>/`):
```
runs.csv            one row per run (model × harness × condition × delivery × budget × repeat, over one task set)
tasks.csv           full grid: one row per (run, task), INCLUDING not_executed tasks
model_calls.csv     one row per model call
tool_calls.csv      one row per tool call (our addition: raw vs delivered, evidence hits)
trajectories/archive.jsonl.zst + trajectory_index.csv (run_id, task_id, n_steps, record_index, record_offset)
trajectories/selected/*.json   (auto-picked exemplars: failures by category, best/worst)
raw_api/<run_id>/<task_id>/*.json   full provider requests/responses (replayable)
manifest.json       campaign config, price catalog, runner version, environment
```
- **runs.csv**: `run_id, campaign_id, benchmark_id, model_name` (canonical ID from a registry, never free text), `provider, harness_name` (enum: `direct_tool`, `planner_evaluator`, …), `condition_id, delivery_policy, effort, search_backend, token_budget_per_task, step_cap, repeat_index, repeat_group_id, config_hash` (hash of every setting that defines the setup, so repeats are identified by config, not by a launch label), `runner_version` (package version + git SHA, mandatory), `price_catalog_version` (mandatory), `seed, run_status` (`finished|partial|aborted`), `tasks_expected, tasks_executed, tasks_passed, total_cost_usd, prompt_tokens` (uncached), `cached_tokens, cache_write_tokens, completion_tokens, started_at, ended_at`. No stored score column; scores are derived.
- **tasks.csv**: `run_id, task_id, topic/difficulty` (if available), `task_status` (`completed|not_executed|errored`), `passed` (null if not executed), `exception_count, error, call_count, tool_call_count`, token columns, `cost_usd, latency_seconds, final_phase, termination_reason` (`final_answer|step_cap|token_cap|time_cap|cost_cap|refusal|exception`), `extracted_answer, confidence, grader_verdict, grader_model`, plus evidence-funnel flags (`retrieved_hit, delivered_hit, attended`) and `failure_category`.
- **model_calls.csv**: `call_id, run_id, task_id, call_index, harness_phase` (`planning|execution|evaluation|recovery|finalization`), `prompt_tokens, cached_tokens, cache_write_tokens, completion_tokens, stop_reason, latency_seconds, exception_type, total_cost_usd, n_tool_calls_emitted`.
- **Trajectory record**: one step per model call, in the same order, phase and usage as `model_calls.csv`, followed by a closing step with no usage (final answer, grade, termination reason). Each step has `step, harness_phase, reasoning` (thinking summary/progress notes, where available), `tool, tool_input, tool_output_raw, tool_output_delivered, delivery_meta` (policy, raw_tokens, delivered_tokens, truncated, chunk_index), `raw_response`, usage, `latency_seconds, exception_type`. It is also emitted as OTel GenAI spans (`invoke_agent` → `chat` → `execute_tool`) for optional external viewers.

**Integrity validator** (runs automatically at the end of every run, and as `agent-eval validate <campaign>`), based on the faults found in the example data:
- Rollups reconcile exactly: call cost/tokens sum to the task totals, and task totals sum to the run totals. Every task row has exactly one run, and trajectory steps equal the model calls + 1.
- Missing `runner_version` or `price_catalog_version` stops the run. Costs are always recomputed from the versioned catalog (with the Haiku >100K-prompt tier), never from a fallback rate.
- Flags a run as **unverifiable** if it has tasks passed but zero model calls or no trajectories, and as an **infrastructure failure** when most calls failed to parse or raised (excluded from performance analysis, kept for ops analysis).
- Anomaly checks: cached-token share outliers between repeats, name normalisation drift, and partial runs (`tasks_executed < tasks_expected`).

**Scoring conventions** (from the example notebook):
- The headline score is `tasks_passed / tasks_expected`, with not-executed tasks counted as failures. The executed-only rate and the difficulty-reweighted rate are also reported.
- CIs are run-clustered. Two runs are repeats only if their `config_hash` values match.
- Harness and budget comparisons use a crossed, matched design: same model, budget, runner and tasks, with ≥3 repeats.
- Token consumption is treated as an outcome, not a lever.

Tooling: `agent-eval report` / an HTML trace viewer (step timeline, expandable tool I/O, raw-vs-delivered diff with the answer span highlighted) and a `get_trajectory <run_id> <task_id>` CLI equivalent to the example's.

### Phase 3 — Deep failure analysis + strategic experiments
**3a. Tool conditions (composable YAML):**

| ID | Tools | Corpus | Question answered |
|---|---|---|---|
| C0 `closed_book` | final_answer only | — | internal-knowledge floor / contamination signal |
| C1 `search_snippets` | web_search (snippets) | live web | value of ranking/snippets alone |
| C2 `search_fetch` | web_search + web_fetch | live web | standard browsing agent (baseline) |
| C3 `local_docs` | search_docs + read_doc | BrowseComp-Plus | retrieval over a closed, reproducible corpus |
| C4 `search+docs` | C2 + C3 | both | combination effects / source preference |
| C5 `oracle_docs` | read_doc on gold evidence docs | BrowseComp-Plus qrels | reasoning ceiling given perfect retrieval |

**Delivery-policy axis** (applies to every tool): `full`, `truncate_2k`, `truncate_8k`, `truncate_32k`, `paginate_8k` (+ `read_more`), `summarize`. Plus effort axis (low/medium/high) as an optional sweep.

**3b. Evidence-consumption funnel ("did the model consume the full result?")** — every tool call logs `raw` and `delivered`. Using gold answer + aliases (BrowseComp) and gold evidence docids (BrowseComp-Plus):
`retrieved_hit` (evidence in raw result) → `delivered_hit` (survived delivery policy) → `attended` (later reasoning/answer references it) → `correct`.
Each failure is attributed to the first broken link: **retrieval miss / truncation loss / comprehension miss / synthesis-reasoning error / format-grading**. Paginated runs also record whether the agent called `read_more` when the evidence was beyond the first chunk.

**3c. Failure taxonomy & detectors**
- Deterministic signals: cap hit, no/premature final_answer, repeated identical calls (doom loop), query-reformulation churn, tool error rate, empty fetches, truncation events, hedged multi-answers, overconfidence (high confidence + wrong), compaction count, cost outliers.
- LLM labeller (configurable; calibrated on ~50 hand-labelled traces) assigns MAST/TRAIL-derived categories (spec misread, planning, retrieval, tool misuse, evidence ignored, hallucinated fact, verification skipped, premature termination, formatting) + first-error step.
- Metrics: accuracy, pass@k, pass^k, calibration error, 95% task-level bootstrap CIs, $/task, tokens/task, tool calls/task; **paired** condition comparisons on identical question sets (McNemar / paired bootstrap).
- Report: HTML/markdown with failure distributions, attribution funnel per condition × policy, heatmaps, links to exemplar traces.

### Phase 4 — Multi-provider / open-source
`OpenAIClient`, `GeminiClient` (preserve thought signatures), `OpenAICompatibleClient` for vLLM/Ollama (Ollama runs natively on Windows — no Docker), text-protocol fallback parser for models without native tool calling. Provider conformance suite (parallel calls, tool errors, long results, multi-turn thinking round-trip, stop-reason mapping) must pass before any benchmark run. Per-model capability flags; tools/prompts/budgets stay identical.

### Phase 5 — Planner–evaluator loop
`PlannerEvaluatorAgent` (ADK LoopAgent semantics): **Planner** decomposes the question into constraints + search plan + success checklist → **Executor** (Phase 1 ReAct agent, same tools) → **Evaluator** verifies every constraint is backed by a cited source and the answer is single (no hedging) → `accept` or targeted feedback; loop to `max_iterations` (default 3) or budget. Evaluator never sees the gold answer. Variants: same vs stronger evaluator model; evaluator with/without tools. Compare to Strategy 1 in a **crossed, matched design**: same model, token budget, runner version and question set, ≥3 repeats each. Report the paired difference with run-clustered CIs, cost per task and cost per success, and calls per `harness_phase`, and re-run the Phase 3 funnel to see which failure classes it fixes. This avoids the harness/budget confounding seen in the example campaign.

### Later — APEX-Agents (Corporate Lawyer) & APEX-Accounting
Needs Mercor's Harbor runtime (Docker) and gated HF access; we'd add `benchmarks/apex_harbor/` + `tools/mcp_bridge.py` reusing our agent, tracing and analysis unchanged. Not in current scope.

---

## 6. Verification
- Unit tests (`pytest`): decrypt round-trip, grader regex parse, delivery policies (truncation boundaries; pagination reconstructs full content exactly), evidence-hit detection, pass@k/pass^k math, span-tree well-formedness, cost computation vs pricing table (incl. >100K tier).
- Pilot: 5 BrowseComp questions under C0/C2/C3 → inspect traces in viewer, verify grades by hand, confirm logged usage matches API usage, `cache_read_input_tokens` > 0 after turn 1.
- Resume test: kill a run mid-way, resume, verify no duplicated/missing trials; the killed run must show `run_status=partial` with `not_executed` rows until resumed.
- Logging conformance: `agent-eval validate` passes on the pilot campaign. Fault-injection tests confirm the validator catches each example-data fault: a missing price catalog, a run with zero calls, an all-parse-failure run, mismatched rollups, and a trajectory/call count mismatch.
- Reproduce the example notebook's Q1-style score computation on our own `runs.csv`/`tasks.csv` with pandas to confirm the schema supports that analysis.
- Sanity: C5 (oracle docs) accuracy ≫ C0; truncate_2k ≤ full on evidence `delivered_hit`.
