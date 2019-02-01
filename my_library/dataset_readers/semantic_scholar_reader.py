from typing import Dict
import tqdm
import json

from allennlp.common.file_utils import cached_path

from allennlp.data.instance import Instance

from allennlp.data.fields import LabelField, TextField
from allennlp.data.tokenizers import Tokenizer, WordTokenizer
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.dataset_readers.dataset_reader import DatasetReader


@DatasetReader.register("s2_papers")
class SemanticScholarDatasetReader(DatasetReader):

    def __init__(self,
                 lazy : bool,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None) -> None:
        super(SemanticScholarDatasetReader,self).__init__(lazy=True)
        self._tokenizer = tokenizer or WordTokenizer()
        self._token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}

    def _read(self, file_path):
        with open(cached_path(file_path), 'r') as data_file:
            # logger.info("Reading instances from lines in file at : #s", file_path)
            for line_num, line in enumerate(data_file.readlines()):
                line = line.strip("\n")
                if not line:
                    continue
                paper_json = json.loads(line)
                title = paper_json['title']
                abstract = paper_json['paperAbstract']
                venue = paper_json['venue']
                yield self.text_to_instance(title, abstract, venue)

    def text_to_instance(self, title: str, abstract: str, venue: str = None) -> Instance:
        tokenized_title = self._tokenizer.tokenize(title)
        tokenized_abstract = self._tokenizer.tokenize(abstract)
        title_field = TextField(tokenized_title, self._token_indexers)
        abstract_field = TextField(tokenized_abstract, self._token_indexers)
        fields = {'title': title_field, 'abstract': abstract_field}
        if venue is not None:
            fields['label'] = LabelField(venue)
        return Instance(fields)
