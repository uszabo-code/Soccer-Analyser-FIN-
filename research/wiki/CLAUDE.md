# Soccer Analyzer Wiki — Schema & Conventions

This wiki is a persistent knowledge base for the soccer-analyzer project. It compiles understanding of problems, techniques, and findings that accumulates across sessions. The LLM maintains it; the human curates sources and directs analysis.

---

## Directory Structure

```
research/wiki/
  CLAUDE.md       ← this file (schema)
  index.md        ← catalog of all pages (LLM reads this first on every query)
  log.md          ← append-only chronological record

  problems/       ← known issues with accumulated understanding
  concepts/       ← technical concepts (tracker params, homography, OCR, etc.)
  findings/       ← synthesized conclusions from experiments
  sources/        ← summaries of ingested papers, articles, benchmarks
```

### Relationship to existing research/ structure

- `research/programs/` — active optimization campaigns (parameter search specs). Immutable during a campaign run. The wiki does NOT duplicate these; it cites them.
- `research/experiments/` — raw experiment results. Immutable output. The wiki synthesizes these into `findings/` pages.
- `research/wiki/` — the compiled knowledge layer. LLM writes and maintains it.

---

## Page Formats

### problems/

One page per known issue. Accumulates understanding over time — never overwritten, always extended.

```markdown
---
name: <Short title>
type: problem
severity: critical | significant | minor
status: open | in-progress | resolved
related_programs: [program-name, ...]
related_concepts: [concept-name, ...]
---

## Summary
One paragraph — what the problem is and why it matters.

## Root Cause
Best current understanding. Mark uncertain claims with *(unconfirmed)*.

## Current Understanding
What we know so far. Updated as experiments run.

## What's Been Tried
Bulleted list of approaches and outcomes.

## Open Questions
What we still don't know.

## References
Links to relevant programs, experiments, sources.
```

### concepts/

One page per technical concept relevant to this project. Written for someone who knows the codebase but may not know the literature.

```markdown
---
name: <Concept name>
type: concept
related_problems: [problem-name, ...]
---

## Summary
What it is in 2–3 sentences.

## How It Works
Enough detail to be useful. Diagrams as ASCII if helpful.

## Relevance to This Project
Specifically how it applies here. Include code file references.

## Key Parameters / Variants
If applicable.

## Sources
Papers or articles this is drawn from.
```

### findings/

One page per synthesized conclusion from experiments. These are durable — not raw results, but interpreted takeaways.

```markdown
---
name: <Finding title>
type: finding
date: YYYY-MM-DD
confidence: high | medium | low
related_problems: [problem-name, ...]
related_experiments: [path/to/experiment, ...]
---

## Finding
One clear statement of what was learned.

## Evidence
What experiments or sources support this.

## Implications
What this means for the project — design decisions, next steps.

## Caveats
Conditions under which this finding may not hold.
```

### sources/

One page per ingested source (paper, article, benchmark, experiment summary).

```markdown
---
name: <Title>
type: source
source_type: paper | article | benchmark | video | experiment
date_ingested: YYYY-MM-DD
url: <if applicable>
file: <path if local>
---

## Summary
What the source is about.

## Key Points
Bulleted. Focus on what's relevant to this project.

## Relevance
How it connects to open problems or concepts in this wiki.

## Pages Updated
Which wiki pages were updated when this was ingested.
```

---

## Workflows

### Ingest a source

1. Read the source (or have it provided).
2. Discuss key takeaways with the user.
3. Write a `sources/` page.
4. Create or update relevant `concepts/` pages.
5. Update relevant `problems/` pages (especially "Current Understanding" and "What's Been Tried").
6. If a clear conclusion emerges, write a `findings/` page.
7. Update `index.md` — add new pages, update summaries of changed pages.
8. Append to `log.md`: `## [YYYY-MM-DD] ingest | <source title>`

### Answer a query

1. Read `index.md` to find relevant pages.
2. Read those pages.
3. Synthesize an answer with citations (link to wiki pages, not raw sources).
4. If the answer reveals something worth keeping, offer to file it as a `findings/` page.
5. Append to `log.md`: `## [YYYY-MM-DD] query | <question summary>`

### Lint the wiki

Check for:
- Contradictions between pages
- Stale claims superseded by newer findings
- Orphan pages (no inbound links from index or other pages)
- Concepts mentioned but lacking their own page
- Problems with status=resolved that still appear open elsewhere
- Data gaps worth filling with a web search or new experiment

Append to `log.md`: `## [YYYY-MM-DD] lint | <summary of issues found>`

---

## index.md format

```markdown
# Wiki Index

## Problems
- [Page title](problems/filename.md) — one-line description

## Concepts
- [Page title](concepts/filename.md) — one-line description

## Findings
- [Page title](findings/filename.md) — one-line description

## Sources
- [Page title](sources/filename.md) — one-line description
```

## log.md format

Each entry on its own `##` heading so it's greppable:
```
## [YYYY-MM-DD] <operation> | <title>
Optional one-paragraph note.
```
