"""Generate and audit the synthetic poster collection for laboratory work 4.

Adapted from Irina-64/ml-creative-technologies-course, demo/project.py.
Upstream commit 844a315d71ffc11b8462308e158c24fb16253023.
Copyright (c) 2026 Trira. MIT terms are retained in LICENSE.course.
This adaptation introduces distinct layout families and checks their split.
Python 3.10+ and the standard library are sufficient.
"""

from __future__ import annotations

import csv
from collections import Counter
import hashlib
import json
from pathlib import Path
import re
from xml.etree import ElementTree


ROOT = Path(__file__).resolve().parent
VERSION = "1.1"
GENRES = ["джаз", "театр", "кино", "выставка", "электроника", "фестиваль"]
STYLES = ["минимализм", "детализация"]
SPLITS = ["train", "train", "validation", "validation", "test", "test"]
ALIASES = ["саксофон", "спектакль", "кинопоказ", "галерея", "рейв", "опенэйр"]
COLORS = ["#224B79", "#6A4074", "#B36A36", "#32745E", "#76582E", "#375B6E"]
LAYOUTS = ["круг", "прямоугольники", "треугольник", "линии", "ромб", "эллипсы"]


def digest(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def save_json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def save_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def poster_svg(identifier: str, genre: str, family: int, color: str) -> str:
    """Each family has a different geometry, independent of its printed ID."""
    shapes = [
        '<circle cx="300" cy="420" r="140" fill="COLOR"/>',
        '<rect x="60" y="200" width="120" height="500" fill="COLOR"/>'
        '<rect x="240" y="300" width="120" height="300" fill="COLOR"/>'
        '<rect x="420" y="230" width="120" height="450" fill="COLOR"/>',
        '<polygon points="300,220 80,650 520,650" fill="COLOR"/>',
        '<path d="M60 250 L540 650 M60 350 L540 550 M60 450 L540 450 '
        'M60 550 L540 350 M60 650 L540 250" stroke="COLOR" stroke-width="30"/>',
        '<polygon points="300,200 520,440 300,680 80,440" fill="COLOR"/>',
        '<ellipse cx="300" cy="350" rx="220" ry="70" fill="COLOR"/>'
        '<ellipse cx="300" cy="480" rx="160" ry="60" fill="COLOR"/>'
        '<ellipse cx="300" cy="600" rx="100" ry="50" fill="COLOR"/>',
    ]
    geometry = shapes[family].replace("COLOR", color)
    style = STYLES[family % 2]
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="800" viewBox="0 0 600 800">'
        '<rect width="600" height="800" fill="#FAF7F0"/>'
        f'{geometry}<text x="35" y="65" font-size="24" font-family="Arial">{identifier} / {genre}</text>'
        f'<text x="35" y="105" font-size="18" font-family="Arial">Учебный макет / {style}</text></svg>'
    )


def generate() -> tuple[list[dict], list[dict]]:
    rows = []
    for genre_index, genre in enumerate(GENRES):
        for family in range(6):
            identifier = f"P{genre_index * 6 + family + 1:03d}"
            relative_path = f"data/posters/{identifier}.svg"
            content = poster_svg(identifier, genre, family, COLORS[genre_index]).encode("utf-8")
            path = ROOT / relative_path
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(content)
            rows.append({
                "id": identifier,
                "description": f"{genre} афиша {STYLES[family % 2]} геометрия {LAYOUTS[family]}",
                "genre": genre,
                "style": STYLES[family % 2],
                "author": f"A{family + 1:02d}",
                "series": f"S{family + 1:02d}",
                "layout_family": f"L{family + 1:02d}",
                "group_id": f"A{family + 1:02d}|S{family + 1:02d}|L{family + 1:02d}",
                "split": SPLITS[family],
                "source": "generated:prepare_data.py@1.1",
                "rights_status": "demo_allowed",
                "synthetic": True,
                "data_version": VERSION,
                "width": 600,
                "height": 800,
                "image": relative_path,
                "sha256": digest(content),
            })

    rows.extend(dict(rows[index], id=f"D{index + 1:03d}") for index in range(3))
    rows.extend([
        dict(rows[0], id="X001", rights_status="unknown"),
        dict(rows[8], id="X002", rights_status="unknown"),
        dict(rows[15], id="X003", description=""),
    ])
    queries = []
    for split in ("train", "validation", "test"):
        for genre, alias in zip(GENRES, ALIASES):
            for kind, word, style_word in (
                ("literal", genre, "минимализм"),
                ("synonym", alias, "лаконичный"),
            ):
                queries.append({
                    "id": f"{split}_{len(queries) + 1:02d}",
                    "split": split,
                    "text": f"{word} {style_word}",
                    "genre": genre,
                    "preferred_style": "минимализм",
                    "kind": kind,
                    "source": "synthetic:prepare_data.py@1.1",
                    "synthetic": True,
                })
    return rows, queries


def prepare(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    clean, excluded, seen = [], [], set()
    for row in rows:
        if row["rights_status"] != "demo_allowed":
            reason = "unknown_rights"
        elif not row["description"].strip():
            reason = "empty_description"
        elif row["sha256"] in seen:
            reason = "duplicate_image"
        else:
            seen.add(row["sha256"])
            clean.append(row)
            continue
        excluded.append({"id": row["id"], "reason": reason})
    return clean, excluded


def audit(raw: list[dict], clean: list[dict], queries: list[dict], excluded: list[dict]) -> dict:
    for row in raw:
        assert len(row["id"]) == 4 and re.fullmatch(r"[PDX][0-9]{3}", row["id"])
        assert row["genre"] in GENRES and row["style"] in STYLES
        assert row["split"] in set(SPLITS)
        content = (ROOT / row["image"]).read_bytes()
        assert digest(content) == row["sha256"], row["id"]
        svg = ElementTree.fromstring(content)
        assert int(svg.attrib["width"]) == row["width"] == 600
        assert int(svg.attrib["height"]) == row["height"] == 800
    assert len({row["id"] for row in raw}) == len(raw) == 42
    assert len(clean) == 36 and len(excluded) == 6
    assert Counter(row["reason"] for row in excluded) == {
        "unknown_rights": 2, "empty_description": 1, "duplicate_image": 3,
    }
    assert len({row["sha256"] for row in clean}) == 36
    assert all(row["synthetic"] and row["description"].strip() for row in clean)

    split_rows = {split: [row for row in clean if row["split"] == split] for split in set(SPLITS)}
    overlap_checks = []
    for first, second in (("train", "validation"), ("train", "test"), ("validation", "test")):
        for field in ("id", "author", "series", "layout_family", "group_id", "sha256"):
            overlap = {r[field] for r in split_rows[first]} & {r[field] for r in split_rows[second]}
            assert not overlap, (first, second, field)
            overlap_checks.append({"first": first, "second": second, "field": field, "overlap": len(overlap)})
        normalized = []
        for split in (first, second):
            normalized.append({
                digest(re.sub(rb"<text\b[^>]*>.*?</text>", b"", (ROOT / r["image"]).read_bytes()))
                for r in split_rows[split]
            })
        assert not normalized[0] & normalized[1]
        overlap_checks.append({"first": first, "second": second, "field": "svg_without_text", "overlap": 0})

    for split, records in split_rows.items():
        assert len(records) == 12
        assert Counter(r["genre"] for r in records) == dict.fromkeys(GENRES, 2)
        assert Counter(r["style"] for r in records) == dict.fromkeys(STYLES, 6)
        assert sum(query["split"] == split for query in queries) == 12
    assert len({query["id"] for query in queries}) == len(queries) == 36
    assert all(query["preferred_style"] == "минимализм" for query in queries)

    save_csv(ROOT / "results/split_checks.csv", overlap_checks)
    files = ["prepare_data.py", "data/raw.json", "data/clean.json", "data/queries.json", "data/lexicon.json"]
    files.extend(r["image"] for r in clean)
    return {
        "data_version": VERSION,
        "synthetic": True,
        "raw_count": len(raw),
        "clean_count": len(clean),
        "exclusions": dict(Counter(r["reason"] for r in excluded)),
        "genres": dict(Counter(r["genre"] for r in clean)),
        "styles": dict(Counter(r["style"] for r in clean)),
        "splits": {split: len(records) for split, records in sorted(split_rows.items())},
        "queries": dict(Counter(query["split"] for query in queries)),
        "cross_split_checks_passed": len(overlap_checks),
        "hashes": {name: digest((ROOT / name).read_bytes()) for name in files},
    }


def main() -> None:
    raw, queries = generate()
    clean, excluded = prepare(raw)
    save_json(ROOT / "data/raw.json", raw)
    save_json(ROOT / "data/clean.json", clean)
    save_json(ROOT / "data/queries.json", queries)
    save_json(ROOT / "data/lexicon.json", {
        "version": "1.0",
        "source": "course glossary at 844a315d71ffc11b8462308e158c24fb16253023",
        "mapping": dict(zip(ALIASES, GENRES)) | {"лаконичный": "минимализм", "насыщенный": "детализация"},
    })
    save_csv(ROOT / "results/exclusions.csv", excluded)
    summary = audit(raw, clean, queries, excluded)
    save_json(ROOT / "results/audit_summary.json", summary)
    print(f"Data v{VERSION}: {len(raw)} raw -> {len(clean)} clean; {len(queries)} queries.")
    print(f"Cross-split checks passed: {summary['cross_split_checks_passed']}.")


if __name__ == "__main__":
    main()
