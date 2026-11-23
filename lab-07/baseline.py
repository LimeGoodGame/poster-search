"""Run and evaluate the lab 07 lexical TF-IDF baseline. Python 3.10+, stdlib."""
import argparse
from collections import Counter, defaultdict
import csv
from hashlib import sha256
import json
from math import fsum, isclose, log2
from pathlib import Path
import platform
from runpy import run_path

LAB = Path(__file__).resolve().parent
LAB06 = LAB.parent / "lab-06"
RESULTS = LAB / "results"


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def load_model():
    settings_bytes = (LAB / "settings.json").read_bytes()
    settings = json.loads(settings_bytes)
    if (settings["fit_split"], settings["evaluation_split"], settings["top_k"],
            settings["min_relevant_grade"], settings["ndcg_gain"]) != (
            "train", "validation", 3, 1, "2^grade - 1"
    ):
        raise ValueError("Настройки не соответствуют протоколу ЛР № 7")

    manifest_bytes = (LAB06 / "data/manifest.json").read_bytes()
    if sha256(manifest_bytes).hexdigest() != settings["manifest_sha256"]:
        raise ValueError("Манифест набора отличается от зафиксированной версии")
    manifest = json.loads(manifest_bytes)
    if manifest["data_version"] != settings["data_version"]:
        raise ValueError("Версия набора не совпадает с настройками")
    for relative in ("prepare.py", "data/clean.json", "data/queries.json", "data/qrels.csv", "data/idf.json"):
        if sha256((LAB06 / relative).read_bytes()).hexdigest() != manifest["sha256"][relative]:
            raise ValueError(f"Изменился входной файл {relative}")

    posters = read_json(LAB06 / "data/clean.json")
    eligible = [poster for poster in posters if poster["rights_status"] == "demo_allowed"]
    train = [poster for poster in eligible if poster["split"] == settings["fit_split"]]
    catalog = [poster for poster in eligible if poster["split"] == settings["evaluation_split"]]
    if len(train) != 12 or len(catalog) != 12 or len({poster["id"] for poster in catalog}) != 12:
        raise ValueError("Ожидалось 12 карточек для расчёта IDF и 12 карточек для validation")

    preprocess = run_path(str(LAB06 / "prepare.py"))
    idf = preprocess["fit_idf"](train)
    reference_idf = read_json(LAB06 / "data/idf.json")
    if (
        (reference_idf["fit_split"], reference_idf["documents"]) != ("train", len(train))
        or idf.keys() != reference_idf["terms"].keys()
        or not all(isclose(value, reference_idf["terms"][term], rel_tol=1e-12)
                   for term, value in idf.items())
    ):
        raise ValueError("IDF не совпал с весами ЛР № 6")
    return settings, sha256(settings_bytes).hexdigest(), Baseline(
        catalog, idf, settings["top_k"], preprocess["tokenize"], preprocess["vectorize"]
    )


class Baseline:
    def __init__(self, catalog, idf, k, tokenize, vectorize):
        self.k = k
        self.tokenize = tokenize
        self.vectorize = vectorize
        self.idf = idf
        self.documents = {
            item["id"]: self.vectorize(item["description"], self.idf, {}) for item in catalog
        }

    def search(self, text):
        tokens = self.tokenize(text)
        query = self.vectorize(text, self.idf, {})
        scores = []
        for poster_id, document in self.documents.items():
            shared = sorted(query.keys() & document.keys())
            score = fsum(query[term] * document[term] for term in shared)
            if score > 0:
                scores.append((score, poster_id, shared))
        scores.sort(key=lambda item: (-item[0], item[1]))

        reason = None
        if not tokens:
            reason = "empty_query"
        elif not query:
            reason = "no_known_terms"
        elif not scores:
            reason = "no_positive_scores"
        return {
            "query": text,
            "known_terms": sorted(query),
            "unknown_terms": sorted(set(tokens) - self.idf.keys()),
            "empty_reason": reason,
            "results": [
                {"rank": rank, "poster_id": poster_id, "score": round(score, 8), "matched_terms": shared}
                for rank, (score, poster_id, shared) in enumerate(scores[:self.k], 1)
            ],
        }


def load_validation(split, poster_ids):
    queries = [
        query for query in read_json(LAB06 / "data/queries.json") if query["split"] == split
    ]
    grades = {query["id"]: {} for query in queries}
    if len(queries) != 12 or len(grades) != 12:
        raise ValueError("Ожидалось 12 уникальных запросов validation")
    if Counter(query["kind"] for query in queries) != {"literal": 6, "synonym": 6}:
        raise ValueError("Ожидалось по 6 буквальных и синонимических запросов")
    with (LAB06 / "data/qrels.csv").open(encoding="utf-8", newline="") as stream:
        for row in csv.DictReader(stream):
            if row["split"] != split:
                continue
            query_id, poster_id = row["query_id"], row["poster_id"]
            grade = int(row["relevance"])
            if (query_id not in grades or poster_id not in poster_ids
                    or poster_id in grades[query_id] or grade not in (0, 1, 2)):
                raise ValueError(f"Недопустимая оценка релевантности для {query_id} и {poster_id}")
            grades[query_id][poster_id] = grade
    if any(judgments.keys() != poster_ids for judgments in grades.values()):
        raise ValueError("Для каждого запроса нужна оценка всех карточек каталога")
    return queries, grades


def dcg(grades, k):
    return fsum((2 ** grade - 1) / log2(position + 2) for position, grade in enumerate(grades[:k]))


def ranking_metrics(ranked_grades, catalog_grades, k, relevant_grade):
    catalog_grades = list(catalog_grades)
    ideal = dcg(sorted(catalog_grades, reverse=True), k)
    relevant = sum(grade >= relevant_grade for grade in catalog_grades)
    return (
        dcg(ranked_grades, k) / ideal if ideal else 0.0,
        sum(grade >= relevant_grade for grade in ranked_grades[:k]) / relevant if relevant else 0.0,
    )


def evaluate(model, queries, qrels, settings):
    cases = []
    for query in queries:
        search = model.search(query["text"])
        judgments = qrels[query["id"]]
        for hit in search["results"]:
            hit["grade"] = judgments[hit["poster_id"]]
        ranked = [hit["grade"] for hit in search["results"]]
        ndcg, recall = ranking_metrics(
            ranked, judgments.values(), model.k, settings["min_relevant_grade"]
        )
        cases.append({
            "query_id": query["id"], "kind": query["kind"], "genre": query["genre"],
            "target_style": query["preferred_style"], **search,
            "relevant_in_catalog": sum(grade >= settings["min_relevant_grade"]
                                       for grade in judgments.values()),
            "ndcg_at_3": ndcg, "recall_at_3": recall,
            "outcome": "success" if ranked[:1] == [2] and recall == 1 else "error",
        })
    return cases


def summarize(cases):
    segments = defaultdict(list)
    for case in cases:
        segments["all"].append(case)
        segments[case["kind"]].append(case)
    rows = []
    for name in ("all", "literal", "synonym"):
        group = segments[name]
        rows.append({
            "segment": name, "queries": len(group),
            "ndcg_at_3": f"{fsum(case['ndcg_at_3'] for case in group) / len(group):.6f}",
            "recall_at_3": f"{fsum(case['recall_at_3'] for case in group) / len(group):.6f}",
            "successes": sum(case["outcome"] == "success" for case in group),
            "errors": sum(case["outcome"] == "error" for case in group),
            "empty_results": sum(not case["results"] for case in group),
        })
    return rows


def save_csv(filename, rows):
    with (RESULTS / filename).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def save_evaluation(cases, settings, settings_hash, model):
    RESULTS.mkdir(exist_ok=True)
    rows = [
        {
            **{field: case[field] for field in ("query_id", "kind", "query", "genre", "target_style")},
            "returned_ids": " ".join(hit["poster_id"] for hit in case["results"]),
            "grades": " ".join(str(hit["grade"]) for hit in case["results"]),
            "ndcg_at_3": f"{case['ndcg_at_3']:.6f}", "recall_at_3": f"{case['recall_at_3']:.6f}",
            "outcome": case["outcome"], "empty_reason": case["empty_reason"] or "",
        }
        for case in cases
    ]
    summary = summarize(cases)
    save_csv("validation_cases.csv", rows)
    save_csv("metrics.csv", summary)
    environment = {
        "python": platform.python_version(), "implementation": platform.python_implementation(),
        "dependencies": "standard library",
        "data_version": settings["data_version"], "data_commit": settings["data_commit"],
        "settings_sha256": settings_hash, "idf_terms": len(model.idf),
        "fit_split": settings["fit_split"], "evaluation_split": settings["evaluation_split"],
        "top_k": settings["top_k"], "random_seed": None,
    }
    for filename, payload in (("validation_detail.json", cases), ("environment.json", environment)):
        (RESULTS / filename).write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary[0]


def main():
    parser = argparse.ArgumentParser(description="TF-IDF baseline для поиска афиш")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("evaluate", help="Воспроизвести оценку всех запросов validation")
    search = commands.add_parser("search", help="Пробный запрос по каталогу validation")
    search.add_argument("query")
    args = parser.parse_args()
    try:
        settings, settings_hash, model = load_model()
        if args.command == "search":
            print(json.dumps(model.search(args.query), ensure_ascii=False, indent=2))
            return
        queries, qrels = load_validation(settings["evaluation_split"], model.documents.keys())
        summary = save_evaluation(evaluate(model, queries, qrels, settings), settings, settings_hash, model)
    except (OSError, ValueError) as error:
        parser.exit(1, f"Ошибка — {error}\n")
    print(f"Validation · {summary['queries']} запросов · nDCG@3 {summary['ndcg_at_3']} · "
          f"Recall@3 {summary['recall_at_3']} · пустых выдач {summary['empty_results']}")


if __name__ == "__main__":
    main()
