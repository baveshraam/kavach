from types import SimpleNamespace

from kavach.corpus import PROTOCOL_V1
from kavach.studio.plan import DEMO_SENTENCES, RECIPES, build_plan, estimate_minutes, fact_question


def facts(*predicates):
    return [SimpleNamespace(predicate=p) for p in predicates]


def count(items, kind):
    return sum(i.repeat for i in items if i.kind == kind)


def test_every_protocol_prompt_is_in_the_plan() -> None:
    items = build_plan("S1", facts("hometown"))
    free = {i.prompt_id for i in items if i.kind == "free"}
    assert free == {p.prompt_id for p in PROTOCOL_V1}


def test_session_recipes_set_the_repetitions() -> None:
    s1 = build_plan("S1", facts("hometown", "college"))
    r = RECIPES["S1"]
    assert count(s1, "read") == r.read * len(DEMO_SENTENCES)
    assert count(s1, "fact") == r.fact * 2
    s4 = build_plan("S4", facts("hometown"))
    assert count(s4, "read") == RECIPES["S4"].read * len(DEMO_SENTENCES)


def test_an_unknown_session_label_gets_the_default_recipe() -> None:
    assert count(build_plan("X9", []), "read") > 0


def test_each_fact_gets_a_question_and_no_facts_means_no_fact_items() -> None:
    items = build_plan("S1", facts("hometown"))
    fact = [i for i in items if i.kind == "fact"]
    assert len(fact) == 1 and fact[0].text_ta == fact_question("hometown")
    assert not [i for i in build_plan("S1", []) if i.kind == "fact"]


def test_an_unknown_predicate_still_gets_a_question() -> None:
    assert "zodiac" in fact_question("zodiac")


def test_sentences_are_non_empty_and_unique() -> None:
    ids = [s[0] for s in DEMO_SENTENCES]
    assert len(set(ids)) == len(ids) == 3
    assert all(s[1].strip() and s[2].strip() for s in DEMO_SENTENCES)


def test_the_estimate_is_positive_and_grows_with_repeats() -> None:
    assert 0 < estimate_minutes(build_plan("S3", facts("hometown"))) < estimate_minutes(
        build_plan("S1", facts("hometown", "college"))
    )


def test_there_are_two_held_out_sessions_so_the_session_interval_can_mean_something() -> None:
    """One held-out sitting is one cluster; the false-reject interval needs at least two."""
    assert "S4" in RECIPES and "S5" in RECIPES
