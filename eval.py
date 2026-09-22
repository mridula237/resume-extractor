import json
import os
import time
from pathlib import Path

from extract import extract
from schema import Resume

EVAL_DIR = Path("eval")
RESUMES_DIR = EVAL_DIR / "resumes"
GOLD_DIR = EVAL_DIR / "gold"
RESULTS_DIR = EVAL_DIR / "results"
RESULTS_DIR.mkdir(exist_ok=True)

PRICES = {
    "claude-haiku-4-5-20251001": (1.0, 5.0),
    "gpt-5.4-nano": (0.20, 1.25),
}


def cost(model, in_tok, out_tok):
    p_in, p_out = PRICES.get(model, (0, 0))
    return in_tok / 1e6 * p_in + out_tok / 1e6 * p_out


def normalize(val):
    if val is None:
        return ""
    return str(val).strip().lower()


def score_field(pred, gold, field):
    """Returns 1.0 if field matches, 0.0 if not, or partial for lists."""
    p = pred.get(field)
    g = gold.get(field)

    # list fields: jaccard similarity
    if isinstance(g, list):
        if not g and not p:
            return 1.0
        g_set = set(normalize(x) for x in g)
        p_set = set(normalize(x) for x in (p or []))
        if not g_set and not p_set:
            return 1.0
        intersection = g_set & p_set
        union = g_set | p_set
        return len(intersection) / len(union) if union else 1.0

    # nested list fields (education, employment)
    if field in ("education", "employment"):
        if not g:
            return 1.0 if not p else 0.0
        if not p:
            return 0.0
        # check count match and first-entry key fields
        count_score = 1.0 if len(p) == len(g) else 0.5
        key = "institution" if field == "education" else "company"
        g_keys = set(normalize(x.get(key, "")) for x in g)
        p_keys = set(normalize(x.get(key, "")) for x in p)
        overlap = len(g_keys & p_keys) / max(len(g_keys), 1)
        return (count_score + overlap) / 2

    # scalar fields
    return 1.0 if normalize(p) == normalize(g) else 0.0


FIELDS = ["name", "emails", "phones", "location", "total_years_experience",
          "education", "employment", "skills", "certifications"]


def evaluate_one(resume_path, gold_path, model):
    stem = Path(resume_path).stem
    out_path = RESULTS_DIR / f"{stem}_{model}.json"

    if out_path.exists():
        with open(out_path) as f:
            return json.load(f)

    with open(gold_path) as f:
        gold = json.load(f)

    try:
        result = extract(str(resume_path), model)
        pred = result["data"]
        scores = {f: score_field(pred, gold, f) for f in FIELDS}
        record = {
            "stem": stem,
            "model": result["model"],
            "scores": scores,
            "avg_score": sum(scores.values()) / len(scores),
            "in": result["in"],
            "out": result["out"],
            "latency": result["latency"],
            "cost": cost(result["model"], result["in"], result["out"]),
            "error": None,
        }
    except Exception as e:
        record = {
            "stem": stem, "model": model, "scores": {f: 0 for f in FIELDS},
            "avg_score": 0, "in": 0, "out": 0, "latency": 0, "cost": 0,
            "error": str(e),
        }

    with open(out_path, "w") as f:
        json.dump(record, f, indent=2)
    return record


def run_eval(models=("claude", "gpt")):
    pairs = []
    for rf in sorted(RESUMES_DIR.iterdir()):
        stem = rf.stem
        gf = GOLD_DIR / f"{stem}.json"
        if gf.exists():
            pairs.append((rf, gf))

    all_results = {m: [] for m in models}
    total = len(pairs)

    for i, (rf, gf) in enumerate(pairs):
        for model in models:
            print(f"[{i+1}/{total}] {model}: {rf.name}")
            r = evaluate_one(rf, gf, model)
            all_results[model].append(r)
            if r["error"]:
                print(f"  ERROR: {r['error']}")

    return all_results


def write_report(all_results):
    lines = ["# Resume Extraction Evaluation Report\n"]
    lines.append(f"**Resumes evaluated:** {len(list(all_results.values())[0])}  ")
    lines.append(f"**Models:** {', '.join(all_results.keys())}  \n")

    lines.append("## Summary\n")
    lines.append("| Model | Avg Accuracy | Avg Latency | Total Cost | Errors |")
    lines.append("|---|---|---|---|---|")

    for model, results in all_results.items():
        ok = [r for r in results if not r["error"]]
        avg_acc = sum(r["avg_score"] for r in ok) / max(len(ok), 1)
        avg_lat = sum(r["latency"] for r in ok) / max(len(ok), 1)
        total_cost = sum(r["cost"] for r in results)
        errors = sum(1 for r in results if r["error"])
        lines.append(f"| {model} | {avg_acc:.1%} | {avg_lat:.2f}s | ${total_cost:.4f} | {errors} |")

    lines.append("\n## Per-Field Accuracy\n")
    lines.append("| Field | " + " | ".join(all_results.keys()) + " |")
    lines.append("|---|" + "---|" * len(all_results))

    for field in FIELDS:
        row = f"| {field} |"
        for model, results in all_results.items():
            ok = [r for r in results if not r["error"]]
            avg = sum(r["scores"].get(field, 0) for r in ok) / max(len(ok), 1)
            row += f" {avg:.1%} |"
        lines.append(row)

    lines.append("\n## Per-Resume Scores\n")
    lines.append("| Resume | " + " | ".join(all_results.keys()) + " |")
    lines.append("|---|" + "---|" * len(all_results))

    stems = [r["stem"] for r in list(all_results.values())[0]]
    for stem in stems:
        row = f"| {stem} |"
        for model, results in all_results.items():
            match = next((r for r in results if r["stem"] == stem), None)
            if match and not match["error"]:
                row += f" {match['avg_score']:.1%} |"
            else:
                row += " ERROR |"
        lines.append(row)

    report = "\n".join(lines)
    with open("eval/report.md", "w") as f:
        f.write(report)
    print("\nReport saved to eval/report.md")
    return report


if __name__ == "__main__":
    print("Running evaluation on 30 resumes x 2 models...\n")
    all_results = run_eval(models=["claude", "gpt"])
    report = write_report(all_results)
    print("\n" + report)