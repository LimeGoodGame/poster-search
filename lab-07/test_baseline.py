import unittest

from baseline import Baseline, evaluate, load_model, load_validation, ranking_metrics


class BaselineTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.settings, _, cls.model = load_model()

    def test_empty_and_unknown_queries(self):
        for query, reason in [("  … ", "empty_query"), ("саксофон насыщенный", "no_known_terms")]:
            with self.subTest(reason=reason):
                result = self.model.search(query)
                self.assertEqual(result["empty_reason"], reason)
                self.assertEqual(result["results"], [])

    def test_equal_scores_break_ties_by_id_and_keep_top_three(self):
        catalog = [{"id": key, "description": "джаз"} for key in ("P004", "P002", "P003", "P001")]
        catalog.append({"id": "P000", "description": "кино"})
        model = Baseline(catalog, self.model.idf, 3, self.model.tokenize, self.model.vectorize)
        results = model.search("ДЖАЗ")["results"]
        self.assertEqual([hit["poster_id"] for hit in results], ["P001", "P002", "P003"])

    def test_metrics_use_ideal_full_catalog_and_exponential_gain(self):
        ndcg, recall = ranking_metrics([1, 2, 0], [0, 1, 2, 0], 3, 1)
        self.assertAlmostEqual(ndcg, 0.79670758)
        self.assertEqual(recall, 1)
        ndcg, recall = ranking_metrics([1, 0, 0, 2], [0, 1, 2, 0], 3, 1)
        self.assertAlmostEqual(ndcg, 0.27541155)
        self.assertEqual(recall, 0.5)

    def test_all_validation_queries_use_only_validation_catalog(self):
        queries, qrels = load_validation(self.settings["evaluation_split"], self.model.documents.keys())
        cases = evaluate(self.model, queries, qrels, self.settings)
        self.assertEqual(len(cases), 12)
        self.assertEqual(sum(case["outcome"] == "success" for case in cases), 6)
        self.assertEqual(sum(case["empty_reason"] == "no_known_terms" for case in cases), 6)
        for case in cases:
            self.assertTrue(case["query_id"].startswith("validation_"))
            self.assertTrue(all(hit["poster_id"] in self.model.documents for hit in case["results"]))


if __name__ == "__main__":
    unittest.main()
