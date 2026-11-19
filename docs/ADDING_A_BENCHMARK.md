# Adding a benchmark

The harness is benchmark-neutral. A benchmark is a small plugin class. The agent loop, tools,
budgets, logging, validation, failure analysis, reports and ablations work for any benchmark
without changes.

## Option 1: no code (question/answer data)

If your data is questions with short answers, use the built-in `jsonl_qa` benchmark:

```jsonl
{"id": "q1", "question": "What is the capital of Australia?", "answer": "Canberra", "aliases": ["Canberra, ACT"], "topic": "geo"}
```

```yaml
benchmark:
  id: jsonl_qa
  params:
    path: path/to/your.jsonl
    grader: exact_match        # or llm_judge for free-form answers
```

Extra fields (`topic` above) become `meta_<field>` columns in `tasks.csv`, so you can slice
results by them. See `src/agent_eval/benchmarks/jsonl_qa.py` for all parameters: field names,
numeric tolerance, an optional local corpus with evidence doc ids, and leak patterns.
`configs/experiments/example_jsonl.yaml` runs a 6-question sample.

## Option 2: a plugin class

Copy `src/agent_eval/benchmarks/_template.py`. Only two methods are required:

```python
from agent_eval.benchmarks import Benchmark, Evidence, Task, register
from agent_eval.graders import exact_match

@register("my_bench")
class MyBench(Benchmark):
    def load_tasks(self) -> list[Task]:
        return [Task(task_id="ex-1", prompt="...", reference="gold", evidence=Evidence(answer_strings=["gold"]))]

    async def grade(self, task, response, answer, judge):
        return exact_match((answer or {}).get("answer", ""), [task.reference])
```

Make it available in one of two ways:
- Register it, and import the module in `benchmarks/base.py::_import_builtins`.
- Skip registration and reference it by import path: `benchmark: {id: "my_pkg.my_module:MyBench"}`.

Check it's picked up with `agent-eval benchmarks`.

### What the harness does vs what the plugin does

| Concern | Harness (generic) | Plugin hook (optional unless marked) |
|---|---|---|
| Data | sampling (`num_examples`, `seed`, `task_ids`), resume, task grid | **`load_tasks()`**, `prepare()` |
| What the agent sees | agent loop, budgets, nudges, tool execution | `base_prompt()`, `task_prompt()`, `answer_tool()` |
| Tools | tool registry, toolsets, delivery policies, leak blocklist | `toolsets()`, `corpus_dir()`, `leak_patterns`, `oracle_doc_ids()` |
| Grading | judge model and pricing, `grading_error` handling | **`grade()`**, `format_response()`, `confidence()` |
| Analysis | CSVs, trajectories, validator, funnel, signals, labeller, reports | `Task.evidence`, `Task.metadata` |

### The types

- **`Task`**:
  - `task_id`: stable across runs, used for resume and paired comparisons.
  - `prompt`: what the agent sees.
  - `reference`: the gold answer or grading spec. **Never shown to the agent.**
  - `metadata`: becomes `meta_*` columns.
  - `evidence`: optional (see below).
- **`Evidence`**: `answer_strings` (strings whose presence in a tool output means the answer was retrieved) and/or `doc_ids` (corpus documents that hold the evidence). It powers the retrieved → delivered → attended funnel. Without it, failures are reported as `unattributed_failure`.
- **`GradeResult`**:
  - `passed`: binary outcome, used for pass rates and pass@k / pass^k.
  - `score`: optional value in [0, 1], e.g. the fraction of rubric criteria met.
  - `details`: become `grade_<key>` columns. A `parse_ok` key is checked by the validator.
  - Set `passed=None` when *grading* failed, not the agent. The trial is then logged as `grading_error`, not as a wrong answer.

### Graders you can reuse (`agent_eval.graders`)

| Grader | Use for |
|---|---|
| `exact_match(answer, refs, numeric_tolerance=...)` | short factual answers, free and deterministic |
| `TemplateJudge(template, verdict_re, extract)` | an LLM judge with your own prompt. BrowseComp's official grader is one instance. |
| `grade_rubric(judge, task, response, criteria)` | binary rubric criteria, APEX-style. score = fraction met; passed = all met. |
| `judge.ask(prompt)` | anything custom. Priced and retried, and the model is set by `grader:` in the YAML. |

### Tools and toolsets

A toolset is a named list of tools. The run's `condition` picks one.
- **Built-in toolsets** (`agent-eval toolsets`): `closed_book`, `search_snippets`, `search_fetch`, `local_docs`, `search_docs`, `oracle_docs`.
- **Adding or overriding toolsets:**
  - in a plugin, with `toolsets()`
  - in an experiment, without code:
    ```yaml
    toolsets:
      docs_only: [search_docs, read_doc]
    matrix:
      condition: [closed_book, docs_only]
    ```
- **New tools:** decorate a factory with `@register_tool("name")` in `agent_eval.tools`. It receives a `ToolBuildContext` (price catalog, search backend, corpus, oracle doc ids, blocklist).
- **Local corpus:** return a directory from `corpus_dir()` containing `corpus.jsonl` (`{"docid", "text", "url"?, "title"?}` per line). Build its BM25 index once with `agent_eval.tools.local_docs.Corpus.build_index(dir)`, typically in `prepare()`.

### Checklist before a real campaign
1. `agent-eval prepare <id or experiment.yaml>` loads the expected number of tasks.
2. Run a few tasks offline with the mock provider (see `tests/test_end_to_end.py::IslandBench`) and check `agent-eval validate` passes.
3. Hand-check about 10 grades against the reference before trusting the grader.
4. Set `leak_patterns` to any URL substrings that would reveal your answers. Trials that hit them are flagged as leak suspects and excluded from scores.
