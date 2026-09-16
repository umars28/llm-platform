# LLM Eval Harness

A regression gate for LLM systems. Deterministic assertions first, an LLM judge
only where they cannot reach, cost treated as a correctness property, and a CI
job that fails the build rather than printing a dashboard.

Built against [ops-copilot](https://github.com/umars28/ops-copilot), whose 30
incident scenarios and recorded traces are the corpus.

## The position

**Most of what people use a judge for does not need one.** Whether a required
concept appears, whether a forbidden claim was made, whether the output parses,
which tools were called, what it cost, how long it took — all decidable in code,
for free, with zero variance between runs.

That last property is the real argument. A judge that varies run to run makes it
impossible to tell whether a score moved because the system changed or because
the judge did, and a judge needs its own evaluation before its verdicts mean
anything. So the judge is left holding only the questions that genuinely require
reading — here, whether an answer joins cause to symptom or merely lists both
and leaves the reader to connect them.

The judge also does not run at all on a case a deterministic check already
failed. Paying a model to elaborate on a decided outcome buys nothing.

## What makes the judge trustworthy enough to gate a merge

**It is versioned.** Model plus rubric form a version string stamped on every
cached verdict and every baseline. If the baseline was graded by
`claude-opus-5/v1` and the run used `v2`, the gate fails as **stale** rather than
as a quality regression, and says to re-baseline. Comparing to numbers a
different grader produced is meaningless, and silently doing it is how eval
suites rot.

**It is cached by content.** The key is a hash of (judge version, question,
subject). An unchanged case is judged once, ever. Re-running a commit costs
nothing and returns identical verdicts.

**It is asked closed questions.** Not "is this good" but "does this name the
causal chain, and if not, what is missing". A judge asked to rate quality
produces a number that drifts; asked about presence, it produces one that can be
checked.

## The gate

Four distinct failures, because they need different responses:

| failure | why it is separate |
| --- | --- |
| a case that passed now fails | a specific regression with a specific cause; never averaged away by a threshold |
| aggregate score drops past tolerance | catches an already-failing case losing further ground |
| cost rises past tolerance | a change that improves quality and triples the bill is a regression to whoever pays |
| baseline graded by a different judge | not a quality problem; resolved by re-baselining |

Two more that fail the build and are easy to omit: a baseline case that **did not
run** (a suite that quietly shrinks stops catching things), and a case with **no
recorded subject** (skipping is indistinguishable from passing).

`llm-eval baseline` refuses to record a run with failing cases unless forced. A
baseline is a claim that this state is correct.

## Cost as a first-class number

Cache reads are counted separately from fresh input, because that separation
*is* the optimisation — a prompt caching change shows up as tokens moving into
the cache-read column at a tenth of the price, and totalling them together hides
exactly the effect being measured.

The caching saving is stated against a **counterfactual** — what the same tokens
would have cost with no cache — rather than against a previous run. Comparing to
an earlier run conflates the cache with everything else that changed between
them, which is how caching wins get overstated.

`UsageLedger` tracks per-model spend so a routing change (cheap model for triage,
expensive model for reasoning) can be reported as a share shift rather than a
single number.

## CI

Two jobs, split deliberately:

**Deterministic** needs no credentials, costs nothing, and blocks the merge on
its own. It runs on forks, where secrets are unavailable — which is exactly when
a repository is most exposed.

**Judged** requires a key and is skipped on fork pull requests. By then the
deterministic job has already caught everything decidable, so skipping it is a
loss of nuance rather than a hole.

The judge cache is keyed on the rubric, so a rubric change starts a fresh cache
instead of silently reusing verdicts from a different grader.

## Running it

```bash
uv venv --python 3.12 && uv pip install -e ".[dev]"

llm-eval run --no-judge          # deterministic only, no credentials needed
llm-eval run                     # adds the judge
llm-eval gate --no-judge         # compare to the committed baseline
llm-eval baseline                # record the current run as correct
```

Traces are ops-copilot run records dropped into `traces/`. Scoring a recorded
artefact rather than re-running the agent is what lets the deterministic half
run in CI for free, and what makes a suite re-scorable after a rubric change
without paying for the whole run again.

`pytest` runs 83 tests, none of which need an API key —
`tests/test_gate.py` is the one worth reading first.

## Status

The deterministic half runs against real ops-copilot traces today. Four cases
with recorded traces pass; two fail because their traces do not exist yet, which
is the designed behaviour -- a case with no subject fails loudly rather than
being skipped, because skipping is indistinguishable from passing.

The first real run also caught something worth catching: SC-001 satisfied every
content assertion and failed the **latency budget** at 164s against 120s. The
diagnosis was right and the run was too slow, which is exactly the split this
harness exists to make visible. The budget has since been raised to 180s to
reflect what a correct run on a slow model actually takes, rather than the
number that was guessed before any run existed.

The judged half has not run: it needs API credit. No judged numbers are quoted
here until it has.

## Where it fits

The fourth of four projects that share a spine:
[ops-copilot](https://github.com/umars28/ops-copilot) diagnoses incidents
through MCP tools, [runbook-rag](https://github.com/umars28/runbook-rag)
retrieves documentation into its context,
[agent-guardrail](https://github.com/umars28/agent-guardrail) defends the
context that retrieval and tool use expose, and this measures whether any of it
is getting better or just changing.

## Licence

MIT
