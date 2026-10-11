"""The paper's promise: every result in it is generated. No typed numbers, no undefined macros, a refusal when a results file is missing."""

import re
from pathlib import Path

import pytest

from kavach.eval.paper_build import BuildError, main, num, pct, word, write_macros

ROOT = Path(__file__).resolve().parents[1]
MAIN = ROOT / "paper" / "ieee" / "main.tex"
REPORT = ROOT / "paper" / "report" / "main.tex"
NUMBERS = ROOT / "paper" / "ieee" / "generated" / "numbers.tex"
needs_paper = pytest.mark.skipif(not (MAIN.exists() and NUMBERS.exists()), reason="the paper has not been built (run_paper.ps1)")
MANUSCRIPTS = [m for m in (MAIN, REPORT) if m.exists()]
ids = [m.parent.name for m in MANUSCRIPTS]

#: Protocol constants that are written in the text because they are choices, not results.
ALLOWED_DECIMALS = {"0.05", "0.1", "0.55", "0.62"}  # P_target, a rate target, the live voice floor and its default
ALLOWED_PERCENTS = {"1", "5", "20", "0.1", "60", "95"}  # rate targets, the trimmed-template share, the tail rule, the interval level


def _strip(tex: str) -> str:
    tex = re.sub(r"(?<!\\)%.*", "", tex)  # comments
    tex = tex.split("\\begin{document}")[-1]  # the preamble only holds layout
    tex = re.sub(r"\d*\.?\d+\s*(cm|pt|mm|em|ex|in)\b", "", tex)  # layout lengths: 2.3cm, 0.7pt, \\[0.4cm]
    tex = re.sub(r"\d*\.?\d+\s*\\(textwidth|columnwidth|linewidth)", "", tex)  # width=0.78\textwidth
    tex = re.sub(r"\\(cite|ref|label|input|includegraphics)(\[[^\]]*\])?\{[^}]*\}", "", tex)
    return re.sub(r"\\bibitem\{.*", "", tex.split("\\begin{thebibliography}")[0])


def test_formatting_helpers_spell_digits_and_use_a_real_minus():
    assert word("S1") == "SOne" and word("10") == "OneZero" and word("all") == "All"
    assert pct(0.0049) == "0.49" and num(-0.5, 3) == "$-$0.500"


def test_macro_names_must_be_letters_only(tmp_path):
    with pytest.raises(BuildError, match="letters only"):
        write_macros({"clips1": "3"}, tmp_path / "n.tex")
    write_macros({"clipsOne": "3"}, tmp_path / "n.tex")
    assert "\\newcommand{\\clipsOne}{3}" in (tmp_path / "n.tex").read_text(encoding="utf-8")


def test_the_build_refuses_naming_the_missing_results_file(tmp_path, capsys):
    assert main(["--paper", str(tmp_path)]) == 2
    assert "results.json is missing" in capsys.readouterr().err


@needs_paper
@pytest.mark.parametrize("manuscript", MANUSCRIPTS, ids=ids)
def test_every_generated_macro_the_paper_uses_is_defined(manuscript):
    defined = set(re.findall(r"\\newcommand\{\\(\w+)\}", NUMBERS.read_text(encoding="utf-8")))
    used = set(re.findall(r"\\([A-Za-z]+)", _strip(manuscript.read_text(encoding="utf-8"))))
    ours = re.compile(r"^(poc|pop|spoof|whole|sit|clips|probes|probe[A-Z]|n[A-Z]|owner[A-Z]|seed|nearest)")
    undefined = sorted(m for m in used if ours.match(m) and m not in defined)
    assert not undefined, f"macros used in main.tex but not generated: {undefined}"


@needs_paper
@pytest.mark.parametrize("manuscript", MANUSCRIPTS, ids=ids)
def test_the_paper_has_no_typed_results(manuscript):
    text = _strip(manuscript.read_text(encoding="utf-8"))
    text = re.sub(r"\\(texttt|url)\{[^}]*\}", "", text)
    decimals = {d for d in re.findall(r"(?<![\\\w.])\d+\.\d+", text)} - ALLOWED_DECIMALS
    assert not decimals, f"decimal numbers typed into main.tex (make them macros): {sorted(decimals)}"
    percents = {p for p in re.findall(r"(\d+(?:\.\d+)?)\s*\\?%", text)} - ALLOWED_PERCENTS
    assert not percents, f"percentages typed into main.tex: {sorted(percents)}"


@needs_paper
@pytest.mark.parametrize("manuscript", MANUSCRIPTS, ids=ids)
def test_every_reference_is_cited_and_every_citation_has_a_reference(manuscript):
    tex = manuscript.read_text(encoding="utf-8")
    body, bib = tex.split("\\begin{thebibliography}")
    cited = {k.strip() for grp in re.findall(r"\\cite\{([^}]*)\}", body) for k in grp.split(",")}
    items = set(re.findall(r"\\bibitem\{([^}]*)\}", bib))
    assert cited == items, f"cited but not listed: {sorted(cited - items)}; listed but never cited: {sorted(items - cited)}"
