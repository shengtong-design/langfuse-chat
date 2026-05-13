"""One-shot helper: render SYSTEM_OVERVIEW.md to a styled, single-file HTML.

Usage:
    py -3.12 scripts/md_to_html.py [INPUT.md] [OUTPUT.html]

Defaults to SYSTEM_OVERVIEW.md → SYSTEM_OVERVIEW.html at the repo root.
"""
from __future__ import annotations

import sys
from pathlib import Path

import markdown

REPO = Path(__file__).resolve().parent.parent

CSS = """
:root {
  --fg: #1f2328;
  --muted: #57606a;
  --bg: #ffffff;
  --code-bg: #f6f8fa;
  --border: #d0d7de;
  --accent: #0969da;
  --table-stripe: #f6f8fa;
}
* { box-sizing: border-box; }
html, body { margin: 0; padding: 0; }
body {
  font: 14px/1.55 -apple-system, BlinkMacSystemFont, "Segoe UI", Helvetica, Arial, sans-serif;
  color: var(--fg);
  background: var(--bg);
}
main {
  max-width: 980px;
  margin: 0 auto;
  padding: 48px 56px 96px;
}
h1, h2, h3, h4 {
  margin: 1.6em 0 .6em;
  line-height: 1.25;
  font-weight: 600;
}
h1 { font-size: 2em; border-bottom: 1px solid var(--border); padding-bottom: .3em; margin-top: 0; }
h2 { font-size: 1.5em; border-bottom: 1px solid var(--border); padding-bottom: .3em; }
h3 { font-size: 1.2em; }
h4 { font-size: 1em; color: var(--muted); }
p, ul, ol { margin: .6em 0; }
ul, ol { padding-left: 1.6em; }
li > p { margin: .2em 0; }
a { color: var(--accent); text-decoration: none; }
a:hover { text-decoration: underline; }
hr { border: 0; border-top: 1px solid var(--border); margin: 2em 0; }
code, pre, kbd, samp {
  font-family: "SFMono-Regular", Consolas, "Liberation Mono", Menlo, monospace;
  font-size: 12.5px;
}
code {
  background: var(--code-bg);
  padding: .15em .4em;
  border-radius: 6px;
}
pre {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 18px;
  overflow-x: auto;
  line-height: 1.45;
}
pre code {
  background: transparent;
  padding: 0;
  border-radius: 0;
  font-size: 12.5px;
}
table {
  border-collapse: collapse;
  margin: .8em 0;
  display: block;
  overflow-x: auto;
  max-width: 100%;
}
th, td {
  border: 1px solid var(--border);
  padding: 8px 12px;
  text-align: left;
  vertical-align: top;
}
th { background: var(--code-bg); font-weight: 600; }
tr:nth-child(even) td { background: var(--table-stripe); }
blockquote {
  margin: .8em 0;
  padding: 0 1em;
  color: var(--muted);
  border-left: 3px solid var(--border);
}
.toc {
  background: var(--code-bg);
  border: 1px solid var(--border);
  border-radius: 6px;
  padding: 14px 22px;
  margin: 0 0 24px;
}
.toc ul { margin: .2em 0; padding-left: 1.4em; }
@media (max-width: 720px) {
  main { padding: 24px 18px 64px; }
}
"""

HTML_TEMPLATE = """<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>{title}</title>
<style>{css}</style>
</head>
<body>
<main>
{body}
</main>
</body>
</html>
"""


def render(src: Path, dst: Path) -> None:
    md_text = src.read_text(encoding="utf-8")
    body = markdown.markdown(
        md_text,
        extensions=["fenced_code", "tables", "toc", "sane_lists"],
        extension_configs={"toc": {"title": "Contents", "anchorlink": True}},
        output_format="html5",
    )
    title = src.stem.replace("_", " ").title()
    dst.write_text(
        HTML_TEMPLATE.format(title=title, css=CSS, body=body),
        encoding="utf-8",
    )
    print(f"wrote {dst.relative_to(REPO)} ({dst.stat().st_size:,} bytes)")


def main(argv: list[str]) -> int:
    src = Path(argv[1]) if len(argv) > 1 else REPO / "SYSTEM_OVERVIEW.md"
    dst = Path(argv[2]) if len(argv) > 2 else REPO / "SYSTEM_OVERVIEW.html"
    if not src.is_absolute():
        src = REPO / src
    if not dst.is_absolute():
        dst = REPO / dst
    if not src.exists():
        print(f"error: input not found: {src}", file=sys.stderr)
        return 1
    render(src, dst)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
