from collections import Counter
import csv
import unittest

from prepare import LAB, SPLITS, fit_idf, inspect_style, read_json, verify_split


def sample_svg(geometry):
    return (
        '<svg xmlns="http://www.w3.org/2000/svg" width="600" height="800">'
        '<rect width="600" height="800"/>' + geometry + '<text>чужая метка</text></svg>'
    ).encode()


class PreparationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.posters = read_json(LAB / "data/clean.json")
        cls.queries = read_json(LAB / "data/queries.json")

    def test_idf_ignores_validation_and_test(self):
        original = fit_idf(self.posters)
        altered = [dict(row, description="утечка проверочное слово") if row["split"] != "train" else row
                   for row in self.posters]
        self.assertEqual(fit_idf(altered), original)
        self.assertEqual(read_json(LAB / "data/idf.json")["terms"], original)
        self.assertNotIn("утечка", original)

    def test_split_rejects_a_shared_author_or_query(self):
        verify_split(self.posters, self.queries)
        author = next(row["author"] for row in self.posters if row["split"] == "train")
        with self.assertRaisesRegex(ValueError, "author"):
            verify_split([dict(row, author=author) if row["split"] == "test" else row
                          for row in self.posters], self.queries)
        text = next(row["text"] for row in self.queries if row["split"] == "train")
        with self.assertRaisesRegex(ValueError, "запросов"):
            verify_split(self.posters, [dict(row, text=text.upper() + "!") if row["split"] == "test" else row
                                        for row in self.queries])

    def test_audit_boundary_and_accents(self):
        large = '<rect x="20" y="20" width="60" height="20"/>'
        second = '<rect x="120" y="20" width="60" height="20"/>'
        third = '<rect x="220" y="20" width="60" height="20"/>'
        accent = '<circle cx="400" cy="200" r="10"/>'
        cases = [(large + accent, 1, "минимализм"), (large + second + accent, 2, "review"),
                 (large + second + third, 3, "детализация")]
        for geometry, count, label in cases:
            with self.subTest(label=label):
                observed = inspect_style(sample_svg(geometry))
                self.assertEqual((observed["count"], observed["label"]), (count, label))

    def test_every_query_has_one_match_per_positive_grade_in_its_split(self):
        with (LAB / "data/qrels.csv").open(encoding="utf-8", newline="") as file:
            qrels = list(csv.DictReader(file))
        self.assertEqual(len(qrels), 432)
        lookup = {row["id"]: row["split"] for row in self.posters}
        for split in SPLITS:
            queries = [query for query in self.queries if query["split"] == split]
            self.assertEqual(Counter(query["preferred_style"] for query in queries), {"минимализм": 6, "детализация": 6})
            for query in queries:
                pairs = [row for row in qrels if row["query_id"] == query["id"]]
                self.assertEqual(Counter(row["relevance"] for row in pairs), {"0": 10, "1": 1, "2": 1})
                self.assertEqual(len({row["poster_id"] for row in pairs}), 12)
                self.assertTrue(all(row["split"] == split == lookup[row["poster_id"]] for row in pairs))


if __name__ == "__main__":
    unittest.main()
