# PR Reviewer

An LLM code reviewer with a **measured** recall and false-positive rate, scored
against real defects taken from real git history.

Most LLM PR reviewers are unmeasured. They leave comments, and nobody knows what
share of them are right or how much noise they add. This one carries both
numbers.

The headline result is that an 8B local model finds **0 of 15** real defects
while flagging **20%** of clean diffs — net-negative to ship. That is a finding
about the model at that size, not a limitation of the harness: the harness is
precisely what turns "this reviewer feels noisy" into a number you can act on.

## Results

`qwen3:8b`, running locally, over 55 diffs from four repositories:

| metric | value |
| --- | --- |
| recall — named the real defect | **0%** (0 of 15) |
| detection — flagged anything at all | 13% (2 of 15) |
| false positives on clean diffs | **20%** (8 of 40) |
| findings per clean diff | 0.23 |
| errors | 0 |
| latency | 3s per diff |

**A small local model is not a code reviewer.** It found none of the fifteen
real defects, and commented on one clean diff in five. The two diffs where it
did say something, it said something about unrelated code.

That result held after loosening the scorer. The first keyword set included
filler like "treat" and "running", which would penalise a correct finding phrased
in its own words — so it was tightened to content words only, and recall stayed
at zero. The reviewer's findings are about different things entirely, not the
same things in different words.

Whether a frontier model does better on this corpus is untested; the harness is
model-agnostic and `--backend api` runs it.

## The corpus is the point

Evaluating a reviewer on hand-written buggy snippets measures whether it can
spot a bug someone planted while knowing what they wanted found. That is not the
job.

Here every defect is real: fifteen `fix:` commits from four repositories, each
traced back with `git blame` to the commit that **introduced** the lines the fix
later removed. The defect is genuinely present in the diff under review,
surrounded by working code, with nothing marking it.

Locating it that way is not fussiness. The first version showed the commit
immediately *before* each fix, which seems equivalent and is not — that commit
often never touched the offending lines, so the reviewer would have been asked to
find a defect that was not on screen, and the recall figure would have measured
nothing. A test now asserts that every positive sample actually contains its
defect.

Fixes that only *add* code are excluded. The defect there is an omission, and
"this diff is missing a case it does not mention" is a different and much harder
task; folding it into one recall number would hide which of the two a reviewer
can do.

**Forty clean diffs against fifteen defective ones.** A reviewer that comments on
everything scores perfect recall and is useless, and only clean samples reveal
that. The imbalance is deliberate — most code is fine.

```bash
pr-reviewer extract ~/projects/*        # rebuild from history
pr-reviewer evaluate --backend local    # recall, detection, FPR
pr-reviewer review some.diff            # review one diff
```

## Scoring, and its limits

A finding counts when it names at least half the content words of what the real
fix said. This is crude on purpose: a model judging whether a finding "means the
same thing" as a fix message would need its own evaluation before its verdicts
counted, which is the regress these projects keep running into.

The cost is real and worth stating — a correct finding phrased in entirely
different words can score zero. That is why **detection rate is reported next to
recall**: the gap between them is the reviewer noticing a diff is wrong without
being able to say why, and it bounds how much the keyword scorer might be
underselling.

## A run is only a result if it ran

A run where reviews error out does not report rates. Below 90% completion the
output says `NOT A RESULT` with the completion rate, and the percentages are
marked as describing the survivors.

That check exists because its absence bit immediately. The first evaluation had
26 of 55 reviews fail and duly reported "0% recall" — a number about the 29 that
ran, presented as a number about 55. The cause was worth knowing too: `qwen3` is
a reasoning model whose thinking block consumes the same output budget as the
answer, so on a large diff it spent everything thinking and returned an empty
string. Empty output is now a recorded error with that explanation, not silence
mistaken for "no defects found".

## CI

The workflow comments findings on a pull request and is **advisory** —
`continue-on-error`, never blocking. A reviewer whose false-positive rate is 20%
must not be able to fail a build; that is how a team learns to ignore it, and an
ignored reviewer has a recall of zero whatever it scores here. Make it required
when the measured numbers justify the interruption.

## Where it fits

The fifth of five projects that share a spine:
[ops-copilot](https://github.com/umars28/ops-copilot) diagnoses incidents
through MCP tools, [runbook-rag](https://github.com/umars28/runbook-rag)
retrieves documentation into its context,
[agent-guardrail](https://github.com/umars28/agent-guardrail) defends that
context against injection,
[llm-eval-harness](https://github.com/umars28/llm-eval-harness) gates all of it
on measured quality and cost, and this one reviews the code they are made of —
scored on defects those same repositories actually shipped.

## Licence

MIT
