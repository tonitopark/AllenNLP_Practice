from allennlp.common.testing import ModelTestCase

import my_library

class AcademicPaperClassifierTest(ModelTestCase):
    def setUp(self):
        super(AcademicPaperClassifierTest,self).setUp()
        self.set_up_model('tests/06_b_paper_classifier.json',
                          'tests/s2_papers.json')

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)

