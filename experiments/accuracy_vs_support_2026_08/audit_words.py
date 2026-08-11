"""Check that every word actually RENDERED on a figure is explained in the talk document.

Reads the text off the real figures: it imports make_figures, intercepts each savefig,
and walks every Text artist matplotlib is about to draw. Earlier versions of this check
scanned the source with a regex and missed every panel heading, because those are built
in list variables and never appear inside a set_title(...) call. That bug hid 'ceiling'
and 'readout' from the document for several rounds.

Only figure-to-document is checked. Talk vocabulary that no figure carries is out of
scope here.

The interception delegates to the real savefig, so a run also rewrites all six committed
png files at whatever FIG_DPI is set to.

  uv run --with numpy,scipy,matplotlib,seaborn python audit_words.py
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

# the backend is bound here because this module imports matplotlib before make_figures does
import matplotlib
matplotlib.use("Agg")
from matplotlib.figure import Figure
from matplotlib.text import Text

HERE = Path(__file__).parent
# of the four write-ups that carry these figures TALK is the only one whose prose covers
# every rendered word. MEETING misses three and BRIEF sixteen
DOC = HERE / "TALK_2026_08_18.md"
# the ordinal tick labels 2nd and 10th lose their digits to the word regex below and
# arrive as the bare tails nd, rd and th. the rest are function words with nothing to explain
FREE = {"nd", "rd", "th", "of", "the", "a", "an", "and", "in", "on", "at", "to", "is",
        "it", "no", "or", "be", "so", "as", "by"}

rendered: dict[str, set[str]] = {}
_orig = Figure.savefig


def _capture(self, fname, *a, **k):
    # patched on the class itself, so a plt.savefig call is recorded the same way
    name = Path(str(fname)).name
    seen = set()
    # an unset title or axis label is still a Text artist. fig_where_training_went
    # carries 292 of them and draws 98
    for t in self.findobj(Text):
        s = t.get_text()
        if s and s.strip():
            seen.add(s)
    rendered[name] = seen
    return _orig(self, fname, *a, **k)


Figure.savefig = _capture
# student_curriculum_2026_07 has its own make_figures.py, so this directory has to come
# ahead of whatever the caller's cwd put on the path
sys.path.insert(0, str(HERE))
import make_figures  # noqa: E402

# the __main__ guard does not fire under import, so the entry point is called here
make_figures._main()
# the patch is global to the class, so leaving it installed would keep recording every
# later savefig in the process
Figure.savefig = _orig

# a markdown image tag carries the figure's subject inside its file name, so leaving the
# tags in would let fig_ranking_matters.png explain the word ranking on its own. that word
# is nowhere else in the talk
doc = re.sub(r"!\[[^\]]*\]\([^)]*\)", "", DOC.read_text()).lower()
bad = 0
for fig in sorted(rendered):
    words = set()
    for s in rendered[fig]:
        # tight_layout forces a draw, so the auto tick labels are already filled in when
        # savefig is intercepted. a required leading letter is what keeps those numbers out
        for w in re.findall(r"[A-Za-z][A-Za-z\-']+", s):
            words.add(w.lower())
    # a substring test against the whole document, so a rendered "round" is answered by
    # "rounds" in the prose. it passes more than it should: "go" is covered by "goal" alone
    missing = sorted(w for w in words if w not in FREE and w not in doc)
    flag = "OK " if not missing else "BAD"
    print(f"{flag} {fig:<36} {len(words):>3} words rendered"
          + (f"   MISSING: {', '.join(missing)}" if missing else ""))
    bad += len(missing)

print()
print(f"{'all rendered words are explained' if not bad else str(bad) + ' unexplained words'}")
sys.exit(1 if bad else 0)
