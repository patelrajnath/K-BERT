import unittest

from datautils.constants import mappings_kaggle2conll
from run_kg_luke_ner import normalize_tags


class MyTestCase(unittest.TestCase):
    def test_tag_normalizer(self):
        labels = "O O O O O O O O O O O O O O O O O O B-per_NOKG O O O O O O O O O O O".split()
        labels = "O O O O O B-gpe_NOKG O O O O B-geo O O O O O O O B-gpe_NOKG O O O O O".split()
        n_tags = normalize_tags(labels, mappings_kaggle2conll)
        print(n_tags)
        print(labels)


if __name__ == '__main__':
    unittest.main()
