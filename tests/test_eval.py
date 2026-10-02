import json

from imskos.embed import HashEmbedder
from imskos.eval import e2e, injection, retrieval
from imskos.llm import FakeLLM

from .conftest import DOCS, fake_script

CORPUS = [(src, title, text) for src, (title, text) in DOCS.items()]


def test_question_set_is_well_formed():
    q = retrieval.load_questions()
    assert len(q["answerable"]) == 30 and len(q["web"]) == 10 and len(q["unanswerable"]) == 10
    ids = [x["id"] for k in ("answerable", "web", "unanswerable") for x in q[k]]
    assert len(ids) == len(set(ids))
    for x in q["answerable"]:
        assert x["gold_any"] == [g.lower() for g in x["gold_any"]] and x["answer_any"]
    assert set(injection.QUESTION_IDS) <= {x["id"] for x in q["answerable"]}


def test_metrics_math():
    m = retrieval._metrics([1, 2, None, 5], [[1] + [0] * 9, [0, 1] + [0] * 8, [0] * 10, [0, 0, 0, 0, 1] + [0] * 5])
    assert m["hit@1"] == 0.25 and m["hit@5"] == 0.75 and m["hit@10"] == 0.75
    assert abs(m["mrr@10"] - (1 + 0.5 + 0 + 0.2) / 4) < 1e-9


def test_retrieval_suite_runs_on_tiny_corpus(monkeypatch):
    qs = {"answerable": [{"id": "x", "q": "What does ANNOY search with?", "gold_any": ["random projection trees"], "answer_any": ["x"]}]}
    monkeypatch.setattr(retrieval, "load_questions", lambda: qs)
    r = retrieval.run_retrieval(CORPUS, HashEmbedder(), None)
    assert set(r["modes"]) == {"dense", "bm25", "hybrid"}
    assert r["modes"]["hybrid"]["hit@1"] == 1.0
    json.dumps(r)


def test_e2e_runner_with_fake_llm(monkeypatch):
    qs = {"answerable": [{"id": "a", "q": "What does ANNOY use?", "gold_any": ["random projection trees"], "answer_any": ["random projection"]}],
          "unanswerable": [{"id": "u", "q": "Who founded ANNOY Inc?"}], "web": []}
    monkeypatch.setattr(e2e, "load_questions", lambda: qs)
    from imskos.config import Settings
    eng = e2e.build_engine(CORPUS, FakeLLM(fake_script()), HashEmbedder(), Settings(provider="ollama"))
    res = e2e.run_config("full", eng, ["answerable"], log=lambda *_: None)
    assert res["summary"]["answerable_correct"] == 1.0 and res["summary"]["citation_hit"] == 1.0


def test_injection_runner_defence_blocks_instruction_templates(monkeypatch):
    qs = {"answerable": [{"id": i, "q": "What does ANNOY use to search random projection trees?", "gold_any": ["random projection trees"], "answer_any": ["random projection"]}
                         for i in injection.QUESTION_IDS]}
    monkeypatch.setattr(injection, "load_questions", lambda: qs)
    monkeypatch.setattr(e2e, "load_questions", lambda: qs)

    def obedient(messages, t):                       # a model that follows any instruction it can see
        text = messages[-1]["content"]
        if messages[0]["content"].startswith("You answer"):
            return f"The answer is {injection.CANARY} as stated [1]." if injection.CANARY in text else "ANNOY uses random projection trees [1]."
        return fake_script(grade="[1, 2, 3, 4, 5]")(messages, t)

    from imskos.config import Settings
    res = injection.run_injection(CORPUS, FakeLLM(obedient), HashEmbedder(), Settings(provider="ollama"), log=lambda *_: None)
    s = res["summary"]
    assert s["defense_off"]["instruction_style"]["attack_success_rate"] == 1.0
    assert s["defense_on"]["instruction_style"]["attack_success_rate"] == 0.0
    # content poisoning has no instruction to remove, so the defence does not stop it
    assert s["defense_on"]["content_poisoning"]["attack_success_rate"] > 0.0
