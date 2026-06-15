"""
eval/runner.py — Evaluation runner and report printer
======================================================
Runs the full test suite, prints results, and saves a JSON report.
"""

import json
import sys
import logging
from datetime import datetime
from pathlib import Path

from src.ai.chatbot.scripts.eval.test_cases import TEST_SUITE
from src.ai.chatbot.scripts.eval.scoring import score_response
from src.ai.chatbot.scripts.eval.judge import _llm_judge

BASE_DIR = Path(__file__).parent.parent.parent


def run_evaluation():
    print("=" * 65)
    print("  NEURA Formal Evaluation")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    print("=" * 65)

    # Load pipeline
    print("\nLoading RAG pipeline...")
    try:
        from graph import run_chat
        from crisis import crisis_router
        print("Pipeline loaded OK\n")
    except Exception as e:
        print(f"FAILED to load pipeline: {e}")
        sys.exit(1)

    results     = []
    by_category = {}

    for i, test in enumerate(TEST_SUITE, 1):
        tid      = test["id"]
        category = test["category"]
        query    = test["input"]

        print(f"[{i:02d}/{len(TEST_SUITE)}] {tid} ({category}) — {query[:55]}...")

        # Run crisis check first
        try:
            is_crisis, _ = crisis_router(query)
        except Exception:
            is_crisis = False

        # Run pipeline
        try:
            response, sources, _, _summary = run_chat(
                query       = query,
                history     = [],
                llm_backend = "auto",
            )
        except Exception as e:
            response = f"[ERROR: {e}]"
            sources  = []

        # Automatic rule-based score
        scores = score_response(response, test, is_crisis)
        q      = scores["quality_score"]

        # LLM-as-Judge score
        judge = _llm_judge(query, response)

        result = {
            "id":       tid,
            "category": category,
            "input":    query,
            "response": response,
            "sources":  sources,
            **scores,
            "llm_empathy":     judge["empathy"],
            "llm_naturalness": judge["naturalness"],
            "llm_helpfulness": judge["helpfulness"],
            "llm_safety":      judge["safety"],
            "llm_quality":     judge["llm_quality"],
        }
        results.append(result)

        # Track by category
        by_category.setdefault(category, []).append(q)

        # Quick visual indicator
        bar  = "█" * int(q * 10) + "░" * (10 - int(q * 10))
        flag = "PASS" if q >= 0.6 else "FAIL"
        lq   = judge["llm_quality"]
        lq_str = f"{lq:.2f}" if lq >= 0 else "n/a"
        print(f"      {flag} rule={q:.2f} llm={lq_str} [{bar}] words={scores['word_count']}")
        if not scores["no_banned_opener"]:
            print("      ⚠  banned opener detected")
        if not scores["ends_with_question"] and category not in ("crisis", "edge", "off_topic"):
            print("      ⚠  no closing question")

    # ── Summary ────────────────────────────────────────────────────────────────
    all_scores  = [r["quality_score"] for r in results]
    avg_quality = sum(all_scores) / len(all_scores)
    passed      = sum(1 for s in all_scores if s >= 0.6)

    llm_scores  = [r["llm_quality"] for r in results if r["llm_quality"] >= 0]
    avg_llm     = sum(llm_scores) / len(llm_scores) if llm_scores else -1

    print()
    print("=" * 65)
    print("  RESULTS SUMMARY")
    print("=" * 65)
    print(f"  Total tests      : {len(results)}")
    print(f"  Passed (≥0.6)    : {passed} / {len(results)}  ({100*passed//len(results)}%)")
    print(f"  Avg rule quality : {avg_quality:.3f}")
    print(f"  Avg LLM quality  : {avg_llm:.3f}" if avg_llm >= 0 else "  Avg LLM quality  : n/a")
    print()
    print("  By category:")
    cat_order = ["emotional", "advice", "factual", "off_topic", "crisis", "edge"]
    for cat in cat_order:
        if cat in by_category:
            scores_c = by_category[cat]
            avg_c    = sum(scores_c) / len(scores_c)
            bar      = "█" * int(avg_c * 10) + "░" * (10 - int(avg_c * 10))
            print(f"    {cat:<12} [{bar}]  {avg_c:.2f}  ({len(scores_c)} tests)")

    # Failures
    failures = [r for r in results if r["quality_score"] < 0.6]
    if failures:
        print()
        print("  ⚠  Failed tests:")
        for f in failures:
            print(f"    [{f['id']}] {f['input'][:50]} → quality={f['quality_score']}")

    # Banned opener violations
    violations = [r for r in results if not r["no_banned_opener"]]
    if violations:
        print()
        print("  ⚠  Banned opener violations:")
        for v in violations:
            first_line = v["response"].split(".")[0][:80]
            print(f"    [{v['id']}] {first_line}...")

    # Crisis check
    crisis_tests   = [r for r in results if r["category"] == "crisis"]
    crisis_correct = [r for r in crisis_tests if r["crisis_handled"]]
    print()
    print(f"  Crisis detection: {len(crisis_correct)}/{len(crisis_tests)} correct")

    print()
    print("=" * 65)

    # ── Save report ────────────────────────────────────────────────────────────
    out_dir = BASE_DIR / "evaluation"
    out_dir.mkdir(exist_ok=True)
    out_path = out_dir / f"eval_report_{datetime.now().strftime('%Y%m%d_%H%M')}.json"

    report = {
        "timestamp":   datetime.now().isoformat(),
        "total":       len(results),
        "passed":      passed,
        "avg_quality": round(avg_quality, 3),
        "by_category": {
            cat: round(sum(s) / len(s), 3)
            for cat, s in by_category.items()
        },
        "results": results,
    }

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)

    print(f"\n  Report saved → {out_path.name}")
    print("=" * 65)

    return report
