import multiprocessing
from typing import List, TextIO, Dict
import json
from collections import Counter, OrderedDict, defaultdict, namedtuple
from contextlib import closing
from multiprocessing.pool import Pool

import click
from tqdm import tqdm
from wikipedia2vec.dump_db import DumpDB

PAD_TOKEN = "[PAD]"
UNK_TOKEN = "[UNK]"
MASK_TOKEN = "[MASK]"

SPECIAL_TOKENS = {PAD_TOKEN, UNK_TOKEN, MASK_TOKEN}

Entity = namedtuple("Entity", ["title", "language"])

_dump_db = None  # global variable used in multiprocessing workers


@click.command()
@click.argument("dump_db_file", type=click.Path())
@click.argument("out_file", type=click.Path())
@click.option("--vocab-size", default=1000000)
@click.option("-w", "--white-list", type=click.File(), multiple=True)
@click.option("--white-list-only", is_flag=True)
@click.option("--pool-size", default=multiprocessing.cpu_count())
@click.option("--chunk-size", default=100)
def build_entity_vocab_custom(dump_db_file: str, white_list: List[TextIO], **kwargs):
    dump_db = DumpDB(dump_db_file)
    white_list = [line.rstrip() for f in white_list for line in f]
    EntityVocabCustom.build(dump_db, white_list=white_list, language=dump_db.language, **kwargs)


class EntityVocabCustom(object):
    def __init__(self, vocab_file: str):
        self._vocab_file = vocab_file

        self.vocab: Dict[Entity, int] = {}
        self.counter: Dict[Entity, int] = {}
        self.inv_vocab: Dict[int, List[Entity]] = defaultdict(list)

        # allow tsv files for backward compatibility
        if vocab_file.endswith(".tsv"):
            self._parse_tsv_vocab_file(vocab_file)
        else:
            self._parse_jsonl_vocab_file(vocab_file)

    def _parse_tsv_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r", encoding="utf-8") as f:
            for (index, line) in enumerate(f):
                title, count = line.rstrip().split("\t")
                entity = Entity(title, None)
                self.vocab[entity] = index
                self.counter[entity] = int(count)
                self.inv_vocab[index] = [entity]

    def _parse_jsonl_vocab_file(self, vocab_file: str):
        with open(vocab_file, "r") as f:
            entities_json = [json.loads(line) for line in f]

        for item in entities_json:
            for title, language in item["entities"]:
                entity = Entity(title, language)
                self.vocab[entity] = item["id"]
                self.counter[entity] = item["count"]
                self.inv_vocab[item["id"]].append(entity)

    @property
    def size(self) -> int:
        return len(self)

    def __reduce__(self):
        return (self.__class__, (self._vocab_file,))

    def __len__(self):
        return len(self.inv_vocab)

    def __contains__(self, item: str):
        return self.contains(item, language=None)

    def __getitem__(self, key: str):
        return self.get_id(key, language=None)

    def __iter__(self):
        return iter(self.vocab)

    def contains(self, title: str, language: str = None):
        return Entity(title, language) in self.vocab

    def get_id(self, title: str, language: str = None, default: int = None) -> int:
        try:
            return self.vocab[Entity(title, language)]
        except KeyError:
            return default

    def get_title_by_id(self, id_: int, language: str = None) -> str:
        for entity in self.inv_vocab[id_]:
            if entity.language == language:
                return entity.title

    def get_count_by_title(self, title: str, language: str = None) -> int:
        entity = Entity(title, language)
        return self.counter.get(entity, 0)

    def save(self, out_file: str):
        with open(out_file, "w") as f:
            for ent_id, entities in self.inv_vocab.items():
                count = self.counter[entities[0]]
                item = {"id": ent_id, "entities": [(e.title, e.language) for e in entities], "count": count}
                json.dump(item, f)
                f.write("\n")

    @staticmethod
    def build(
            dump_db: DumpDB,
            out_file: str,
            vocab_size: int,
            white_list: List[str],
            white_list_only: bool,
            pool_size: int,
            chunk_size: int,
            language: str,
    ):
        counter = Counter()
        # count = 0
        with tqdm(total=dump_db.page_size(), mininterval=0.5) as pbar:
            with closing(
                    Pool(pool_size, initializer=EntityVocabCustom._initialize_worker, initargs=(dump_db,))) as pool:
                for ret in pool.imap_unordered(EntityVocabCustom._count_entities, dump_db.titles(),
                                               chunksize=chunk_size):
                    # count += 1
                    # print(count)
                    # if count == 10:
                    #     pool.terminate()
                    #     pool.join()
                    #     break
                    counter.update(ret)
                    pbar.update()

        title_dict = OrderedDict()
        title_dict[PAD_TOKEN] = 0
        title_dict[UNK_TOKEN] = 0
        title_dict[MASK_TOKEN] = 0

        for title_link in white_list:
            if counter[title_link] != 0:
                title_dict[title_link] = counter[title_link]

        if not white_list_only:
            valid_titles = frozenset(dump_db.titles())
            for title_link, count in counter.most_common():
                if title_link in valid_titles and not title_link.startswith("Category:"):
                    title_dict[title_link] = count
                    if len(title_dict) == vocab_size:
                        break

        with open(out_file, "w") as f:
            for ent_id, (title_link, count) in enumerate(title_dict.items()):
                info_box = {}
                if '#' not in title_link and ':' not in title_link and count != 0:
                    info_box = dump_db.get_infobox(title_link)

                json.dump({"id": ent_id, "entities": [[title_link, language]],
                           "count": count, "info_box": info_box}, f, default=str)
                f.write("\n")

    @staticmethod
    def _initialize_worker(dump_db: DumpDB):
        global _dump_db
        _dump_db = dump_db

    @staticmethod
    def _count_entities(title: str) -> Dict[str, int]:
        counter = Counter()
        for paragraph in _dump_db.get_paragraphs(title):
            for wiki_link in paragraph.wiki_links:
                title = _dump_db.resolve_redirect(wiki_link.title)
                counter[title] += 1
        return counter
