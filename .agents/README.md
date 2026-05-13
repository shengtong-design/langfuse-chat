# `.agents/` — skill packs for AI coding agents

Skill packs installed via the [`skills`](https://skills.sh) CLI. Each skill is a
folder with a `SKILL.md` that teaches an agent how to work with a specific
framework or service.

## Layout

```
.agents/
└── skills/
    ├── crewai/                  ← multi-skill vendor: grouped under vendor folder
    │   ├── ask-docs/
    │   ├── design-agent/
    │   ├── design-task/
    │   └── getting-started/
    └── langfuse/                ← single-skill vendor: kept at top level
```

Convention: vendors that ship more than one skill get a subfolder named after
the vendor. Single-skill vendors stay flat for brevity.

## Installed skills

| Skill | Vendor | Source |
|---|---|---|
| `ask-docs` | CrewAI | [crewaiinc/skills](https://github.com/crewaiinc/skills) |
| `design-agent` | CrewAI | [crewaiinc/skills](https://github.com/crewaiinc/skills) |
| `design-task` | CrewAI | [crewaiinc/skills](https://github.com/crewaiinc/skills) |
| `getting-started` | CrewAI | [crewaiinc/skills](https://github.com/crewaiinc/skills) |
| `langfuse` | Langfuse | [langfuse/skills](https://github.com/langfuse/skills) |

`skills-lock.json` at the repo root is the canonical record (vendor, commit
hash, source path). This table is a human-friendly view; if they disagree, the
lock file wins.

## Discovery

The `skills` CLI created Windows directory junctions at `.claude/skills/<name>/`
that point into `.agents/skills/`. Claude Code reads from `.claude/skills/`, so
the actual on-disk content lives once at `.agents/` and is surfaced flatly to
the agent.

## Teammates: how to install

`.agents/` is gitignored. `skills-lock.json` is tracked. To materialize the
same skill set after a clone:

```bash
npx skills experimental_install   # restore from skills-lock.json
```

## Sharp edge: re-grouping after `npx skills add`

The `skills` CLI installs flatly into `.agents/skills/<name>/` and does not
know about the per-vendor subfolder convention. After running `npx skills add
<pkg>` (or `update`), any newly added crewai skill will land at the top level
again and needs to be moved into `.agents/skills/crewai/`, with the
corresponding `.claude/skills/<name>` junction re-pointed.

If this becomes annoying, drop the grouping and go back to flat — the layout
is purely organizational, the CLI does not depend on it.
