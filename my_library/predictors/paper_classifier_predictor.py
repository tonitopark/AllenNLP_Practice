from typing import Tuple
from overrides import overrides

from allennlp.common.util import JsonDict
from allennlp.data import Instance
from allennlp.predictors.predictor import Predictor


@Predictor.register('paper-classifier')
class PaperClassifierPredictor(Predictor):
    """Predictor Wrapper for the AcademicPaperClassifier"""

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        pass

    @overrides
    def predict_json(self, json_dict: JsonDict) -> JsonDict:
        title = json_dict['title']
        abstract = json_dict['paperAbstract']
        instance = self._dataset_reader.text_to_instance(title=title, abstract=abstract)

        label_dict = self._model.vocab.get_index_to_token_vocabulary('labels')

        all_labels = [label_dict[i] for i in range(len(label_dict))]

        return {"instance": self.predict_instance(instance), "all_labels": all_labels}
