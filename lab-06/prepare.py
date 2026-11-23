"""Build lab 06 datasets from the verified lab 04 snapshot. Python 3.10+, stdlib only."""
from collections import Counter
import csv
from hashlib import sha256
from itertools import combinations
import json
from math import hypot, log, sqrt
from pathlib import Path
from random import Random
import re
import runpy
import unicodedata
from xml.etree import ElementTree

LAB = Path(__file__).resolve().parent
REPO = LAB.parent
UPSTREAM = REPO / "lab-04"
DATA_VERSION = "1.2"
GUIDELINE_VERSION = "1.0"
SPLITS = ("train", "validation", "test")
STYLES = ("минимализм", "детализация")
SVG = "{http://www.w3.org/2000/svg}"
MIN_SPAN = 60
POINTS = r"[0-9]+(?:\.[0-9]+)?"
LINE = re.compile(rf"M\s*({POINTS})[ ,]+({POINTS})\s+L\s*({POINTS})[ ,]+({POINTS})")
PATTERNS = {
    "train": (
        "Ищу афишу про «{topic}» в стиле «{style}»",
        "Нужен {style} макет для события {topic}",
    ),
    "validation": (
        "В подборку «{topic}» подойдёт постер в стиле «{style}»",
        "Покажите {style} плакат на тему {topic}",
    ),
    "test": (
        "Выберите плакат из раздела «{topic}» с оформлением в стиле «{style}»",
        "Подберите {style} вариант афиши для рубрики {topic}",
    ),
}


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def save_json(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path, rows, fields):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def digest(payload):
    return sha256(payload).hexdigest()


def style_for_count(count):
    if count == 1:
        return STYLES[0]
    if count >= 3:
        return STYLES[1]
    return "review"


def major_spans(element):
    kind = element.tag.removeprefix(SVG)
    attributes = element.attrib
    if kind == "circle":
        return [2 * float(attributes["r"])]
    if kind == "ellipse":
        return [2 * max(float(attributes[key]) for key in ("rx", "ry"))]
    if kind == "rect":
        return [max(float(attributes[key]) for key in ("width", "height"))]
    if kind == "polygon":
        numbers = list(map(float, re.split(r"[ ,\s]+", attributes["points"].strip())))
        if len(numbers) < 6 or len(numbers) % 2:
            raise ValueError("Некорректные координаты многоугольника")
        x, y = numbers[::2], numbers[1::2]
        return [max(max(x) - min(x), max(y) - min(y))]
    if kind == "path":
        path = attributes["d"]
        parts = list(LINE.finditer(path))
        if not parts or LINE.sub("", path).strip():
            raise ValueError("Поддерживаются только самостоятельные отрезки M…L")
        return [hypot(x1 - x0, y1 - y0) for x0, y0, x1, y1 in
                (map(float, match.groups()) for match in parts)]
    raise ValueError(f"Неподдерживаемый элемент {kind}")


def inspect_style(payload):
    """Independently count geometry, ignoring the background and printed labels."""
    try:
        root = ElementTree.fromstring(payload)
        if root.tag != SVG + "svg" or not len(root):
            raise ValueError("Отсутствует холст SVG")
        background = root[0]
        if not (background.tag == SVG + "rect"
                and background.get("width") == root.get("width")
                and background.get("height") == root.get("height")
                and background.get("x", "0") == background.get("y", "0") == "0"):
            raise ValueError("Первый элемент должен быть фоном всего холста")
        count = sum(
            span >= MIN_SPAN
            for element in list(root)[1:] if element.tag != SVG + "text"
            for span in major_spans(element)
        )
    except (ElementTree.ParseError, ValueError, KeyError) as error:
        return {"count": None, "label": "review", "observation": str(error)}
    return {"count": count, "label": style_for_count(count),
            "observation": f"Число крупных компонентов — {count}"}


def tokenize(text):
    return re.findall(r"[а-яёa-z0-9]+", unicodedata.normalize("NFKC", text).casefold())


def fit_idf(posters):
    train = [row for row in posters if row["split"] == "train"]
    if not train:
        raise ValueError("Обучающий каталог пуст")
    counts = Counter(word for row in train for word in set(tokenize(row["description"])))
    return {word: log((1 + len(train)) / (1 + count)) + 1
            for word, count in sorted(counts.items())}


def vectorize(text, idf, aliases):
    counts = Counter(aliases.get(token, token) for token in tokenize(text))
    weights = {word: counts[word] * idf[word] for word in sorted(counts) if word in idf}
    norm = sqrt(sum(value ** 2 for value in weights.values()))
    return {word: value / norm for word, value in weights.items()} if norm else {}


def verify_split(posters, queries):
    for first, second in combinations(SPLITS, 2):
        a = [row for row in posters if row["split"] == first]
        b = [row for row in posters if row["split"] == second]
        for field in ("id", "author", "series", "layout_family", "group_id", "sha256", "svg_without_text"):
            overlap = {row[field] for row in a} & {row[field] for row in b}
            if overlap:
                raise ValueError(f"Пересечение {field}: {first} и {second}")
        tokens_a = {tuple(tokenize(q["text"])) for q in queries if q["split"] == first}
        tokens_b = {tuple(tokenize(q["text"])) for q in queries if q["split"] == second}
        if tokens_a & tokens_b:
            raise ValueError(f"Одинаковые тексты запросов: {first} и {second}")


def make_queries(genres, aliases):
    inverse = {target: alias for alias, target in aliases.items()}
    queries = []
    for split in SPLITS:
        for genre_index, genre in enumerate(genres):
            for style_index, style in enumerate(STYLES):
                variant = (genre_index + style_index) % 2
                topic, wording = (inverse[genre], inverse[style]) if variant else (genre, style)
                queries.append({
                    "id": f"{split}_{genre_index * 2 + style_index + 1:02d}",
                    "split": split,
                    "text": PATTERNS[split][variant].format(topic=topic, style=wording),
                    "genre": genre, "preferred_style": style,
                    "kind": "synonym" if variant else "literal",
                    "source": "synthetic:lab-06/prepare.py@1.2", "synthetic": True,
                })
    return queries


def make_qrels(posters, queries):
    for query in queries:
        for poster in posters:
            if poster["split"] != query["split"]:
                continue
            relevance = 0
            if poster["genre"] == query["genre"]:
                relevance = 2 if poster["style"] == query["preferred_style"] else 1
            yield {"split": query["split"], "query_id": query["id"], "poster_id": poster["id"],
                   "relevance": relevance, "method": "synthetic_rule"}


def write_pilot(posters, audit, output):
    train = [row for row in posters if row["split"] == "train"]
    Random(23).shuffle(train)
    with (LAB / "pilot_visual.csv").open(encoding="utf-8", newline="") as stream:
        observations = list(csv.DictReader(stream))
    assert len(observations) == len(train) == 12
    svg_rows, final, disagreements = [], [], []
    for number, (poster, visual) in enumerate(zip(train, observations), 1):
        assert visual["blind_id"] == f"B{number:02d}"
        checked = audit[poster["id"]]
        visual_count = int(visual["count"]) if visual["count"].strip() else None
        agree = visual["label"] == checked["label"] and visual_count == checked["count"]
        verdict = checked["label"] if agree else "review"
        svg_rows.append({"blind_id": visual["blind_id"], "poster_id": poster["id"], **checked})
        row = {"blind_id": visual["blind_id"], "poster_id": poster["id"],
               "visual_label": visual["label"], "svg_label": checked["label"],
               "final_label": verdict,
               "decision": "manual_review_required" if verdict == "review" else "count_and_label_agree"}
        final.append(row)
        if verdict == "review":
            disagreements.append({**row, "visual_count": visual["count"], "svg_count": checked["count"],
                                  "basis": checked["observation"]})
    fields = ["blind_id", "poster_id", "visual_label", "svg_label", "final_label", "decision"]
    save_csv(output / "pilot_svg.csv", svg_rows, ["blind_id", "poster_id", "count", "label", "observation"])
    save_csv(output / "pilot_final.csv", final, fields)
    save_csv(output / "disagreements.csv", disagreements, fields + ["visual_count", "svg_count", "basis"])
    agreement = sum(row["decision"] == "count_and_label_agree" for row in final)
    stats = {
        "method": "assistant_visual_review_vs_svg_geometry",
        "independent_human_annotators": 0,
        "split": "train",
        "pilot_cards": len(final),
        "visual_distribution": dict(Counter(row["visual_label"] for row in final)),
        "svg_distribution": dict(Counter(row["svg_label"] for row in final)),
        "final_distribution": dict(Counter(row["final_label"] for row in final)),
        "labels_agree": sum(row["visual_label"] == row["svg_label"] for row in final),
        "counts_and_labels_agree": agreement,
        "agreement_fraction": agreement / len(final),
        "needs_review": len(disagreements),
    }
    save_json(output / "annotation_stats.json", stats)
    return stats


def main():
    source = read_json(UPSTREAM / "results/audit_summary.json")
    assert source["data_version"] == "1.1"
    for path, expected_hash in source["hashes"].items():
        if digest((UPSTREAM / path).read_bytes()) != expected_hash:
            raise ValueError(f"Исходный файл отличается от версии ЛР № 4: {path}")

    upstream_code = runpy.run_path(str(UPSTREAM / "prepare_data.py"))
    raw = read_json(UPSTREAM / "data/raw.json")
    clean, excluded = upstream_code["prepare"](raw)
    assert clean == read_json(UPSTREAM / "data/clean.json")
    assert len(raw) == 42 and len(clean) == 36 and len(excluded) == 6
    results = LAB / "results"
    audit = {row["id"]: inspect_style((UPSTREAM / row["image"]).read_bytes()) for row in clean}
    stats = write_pilot(clean, audit, results)
    if stats["needs_review"] or any(audit[row["id"]]["label"] != row["style"] for row in clean):
        raise ValueError("Обнаружены спорные метки. Проверьте журнал пилота и геометрию каталога.")
    save_csv(results / "exclusions.csv", excluded, ["id", "reason"])

    posters = []
    for row in clean:
        payload = (UPSTREAM / row["image"]).read_bytes()
        normalized = re.sub(rb"<text\b[^>]*>.*?</text>", b"", payload, flags=re.DOTALL)
        posters.append(dict(
            row, image=f"lab-04/{row['image']}", style=audit[row["id"]]["label"],
            svg_without_text=digest(normalized), data_version=DATA_VERSION,
            annotation_method="svg_geometry_rule", annotation_guideline_version=GUIDELINE_VERSION,
            annotation_component_count=audit[row["id"]]["count"],
        ))

    aliases = read_json(UPSTREAM / "data/lexicon.json")["mapping"]
    queries = make_queries(upstream_code["GENRES"], aliases)
    verify_split(posters, queries)
    qrels = list(make_qrels(posters, queries))
    idf = fit_idf(posters)
    data = LAB / "data"
    save_json(data / "clean.json", posters)
    save_json(data / "queries.json", queries)
    save_csv(data / "qrels.csv", qrels, ["split", "query_id", "poster_id", "relevance", "method"])
    train_size = sum(row["split"] == "train" for row in posters)
    save_json(data / "idf.json", {"fit_split": "train", "documents": train_size, "terms": idf})
    save_json(data / "tfidf.json", {
        "posters": {row["id"]: vectorize(row["description"], idf, {}) for row in posters},
        "queries": {row["id"]: {
            "baseline": vectorize(row["text"], idf, {}),
            "with_synonyms": vectorize(row["text"], idf, aliases),
        } for row in queries},
    })
    artifacts = [
        "annotation_guideline.md", "pilot_sheet.png", "pilot_visual.csv", "prepare.py",
        "data/clean.json", "data/queries.json", "data/qrels.csv", "data/idf.json", "data/tfidf.json",
        "results/exclusions.csv", "results/pilot_svg.csv", "results/pilot_final.csv",
        "results/disagreements.csv", "results/annotation_stats.json",
    ]
    save_json(data / "manifest.json", {
        "data_version": DATA_VERSION, "guideline_version": GUIDELINE_VERSION, "synthetic": True,
        "source_version": source["data_version"],
        "source_commit": "00e7ca42ca5e38c91af5fbad18c1b262ef11586c",
        "image_root": "repository",
        "raw_cards": len(raw), "clean_cards": len(posters), "queries": len(queries), "qrels": len(qrels),
        "qrel_distribution": dict(sorted(Counter(row["relevance"] for row in qrels).items())),
        "cards_by_split": dict(Counter(row["split"] for row in posters)),
        "queries_by_split": dict(Counter(row["split"] for row in queries)),
        "annotation": stats,
        "sha256": {path: digest((LAB / path).read_bytes()) for path in artifacts},
    })
    print(f"v{DATA_VERSION}: {len(raw)} → {len(posters)} карточек; {len(queries)} запросов, {len(qrels)} оценок.")
    print(f"Пилот train: {stats['counts_and_labels_agree']}/{stats['pilot_cards']} совпадений, "
          f"{stats['needs_review']} расхождений.")


if __name__ == "__main__":
    main()
