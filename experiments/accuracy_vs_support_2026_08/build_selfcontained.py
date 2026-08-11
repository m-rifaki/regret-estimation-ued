"""Build the two self-contained deliverables from MEETING_2026_08_18.md.

Both outputs embed the six figures as base64, so each is one file that needs nothing
alongside it.

  MCTS-UED-MEETING-<date>.md    reference-style links, base64 at the end of the file
                                   so the prose stays readable as plain text
  MCTS-UED-MEETING-<date>.html  images inlined, styled, renders anywhere

Figures are rendered at FIG_DPI=150 for embedding rather than the 200 used for the
standalone PNGs: 907KB instead of 1281KB, with no loss that matters on screen.

make_figures.py writes next to itself, so the 150 renders have to be moved into embed/ by
hand. That directory is gitignored and starts empty in a fresh checkout.

  FIG_DPI=150 uv run --with numpy,scipy,matplotlib,seaborn python make_figures.py
  uv run --with markdown python build_selfcontained.py [outdir]
"""
from __future__ import annotations

import base64
import re
import sys
from pathlib import Path

HERE = Path(__file__).parent
EMBED = HERE / "embed"
DOCS = {                                   # source markdown -> output stem
    "TALK_2026_08_18.md": "MCTS-UED-2026-08-18",
}
NOTE = (
    "**This file is self-contained.** The six figures are embedded in it directly, so it\n"
    "needs nothing alongside it. If your viewer shows blank boxes instead of figures it is\n"
    "stripping embedded images; open the .html of the same name, which is also\n"
    "self-contained, or the PNGs in `regret-estimation-ued-meeting-2026-08-18/figures/` at full\n"
    "resolution.\n\n")

# the accent is GREEN from make_figures.py so headings and rules match the figures
CSS = """
 body { max-width: 60rem; margin: 3rem auto; padding: 0 1.5rem;
        font: 16px/1.65 -apple-system, "Helvetica Neue", Arial, sans-serif; color: #1a1a1a; }
 h1 { font-size: 2rem; border-bottom: 2px solid #1f6f4a; padding-bottom: .4rem; }
 h2 { font-size: 1.45rem; margin-top: 2.6rem; color: #1f6f4a; }
 h3 { font-size: 1.12rem; margin-top: 1.8rem; }
 img { max-width: 100%; height: auto; display: block; margin: 1.4rem 0;
       border: 1px solid #e3e3e3; border-radius: 4px; }
 table { border-collapse: collapse; margin: 1.2rem 0; font-size: .93rem; }
 th, td { border: 1px solid #d8d8d8; padding: .42rem .7rem; text-align: left; }
 th { background: #f4f7f5; }
 code { background: #f2f2f2; padding: .1rem .32rem; border-radius: 3px; font-size: .9em; }
 pre { background: #f7f7f7; padding: .9rem 1.1rem; border-radius: 5px; overflow-x: auto; }
 pre code { background: none; padding: 0; }
 blockquote { margin: 1.1rem 0; padding: .6rem 1.1rem; border-left: 4px solid #1f6f4a;
              background: #f6faf8; font-weight: 600; }
 hr { border: none; border-top: 1px solid #ddd; margin: 2.6rem 0; }
"""

# figures/ is optional because TALK writes figures/fig_x.png and MEETING writes the bare name
IMG = re.compile(r'!\[([^\]]*)\]\((?:figures/)?(fig_[a-z_]+\.png)\)')


# reads embed/ and never the path in the link, so what goes in is the 150-dpi copy
def b64(name: str) -> str:
    return base64.b64encode((EMBED / name).read_bytes()).decode()


def build_one(src: Path, stem: str, outdir: Path) -> None:
    doc = src.read_text()
    # a source that opens by claiming its numbers are rederived takes the note under that claim
    if "**Every number in this document" in doc:
        doc = doc.replace("**Every number in this document",
                          NOTE + "**Every number in this document", 1)
    else:
        # once only: TALK has a rule between every section and the note belongs above the first
        doc = doc.replace("\n\n---\n", "\n\n" + NOTE + "---\n", 1)

    keys: dict[str, str] = {}
    # the sub records each label as it rewrites, so the tail comes out in figure order
    # [4:-4] drops fig_ and .png, which leaves a label that reads as the figure name
    md = IMG.sub(lambda m: (keys.setdefault(m.group(2)[4:-4], m.group(2)),
                            f"![{m.group(1)}][{m.group(2)[4:-4]}]")[1], doc)
    # NOTE promises six figures and a renamed file would slip past IMG as a link to nothing
    if len(keys) != 6:
        raise SystemExit(f"expected 6 figures, found {len(keys)}: {sorted(keys)}")
    tail = ["", "", "<!-- Figures are embedded below as base64 so this file is self-contained.",
            "     Nothing outside this file is needed to read it. -->", ""]
    for key, fname in keys.items():
        tail += [f"[{key}]: data:image/png;base64,{b64(fname)}", ""]
    (outdir / f"{stem}.md").write_text(md + "\n".join(tail))

    # imported here and not at the top so a missing install still leaves the .md written
    import markdown
    # runs over doc again because md now carries reference labels in place of the paths
    inlined = IMG.sub(
        lambda m: f'![{m.group(1)}](data:image/png;base64,{b64(m.group(2))})', doc)
    body = markdown.markdown(inlined, extensions=["tables", "fenced_code", "toc"])
    (outdir / f"{stem}.html").write_text(
        '<!DOCTYPE html>\n<html lang="en"><head><meta charset="utf-8">\n'
        f"<title>{stem}</title>\n"
        f"<style>{CSS}</style></head><body>\n{body}\n</body></html>")

    # the printed size is the check that the base64 went in: a third above the PNGs it carries
    for ext in ("md", "html"):
        f = outdir / f"{stem}.{ext}"
        print(f"wrote {f.name} ({f.stat().st_size // 1024}KB)")


def build(outdir: Path) -> None:
    for src, stem in DOCS.items():
        build_one(HERE / src, stem, outdir)


if __name__ == "__main__":
    # outdir is never created here so a path given on the command line has to exist already
    build(Path(sys.argv[1]).expanduser() if len(sys.argv) > 1 else Path.home() / "Desktop")
