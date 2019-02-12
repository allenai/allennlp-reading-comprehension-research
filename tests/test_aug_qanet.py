#pylint: disable=unused-import
import pathlib
from allennlp.common.testing import ModelTestCase
from reading_comprehension.drop_models.augmented_qanet import AugmentedQANet
from reading_comprehension.data.drop_reader import DROPReader


class QANetModelTest(ModelTestCase):

    PROJECT_ROOT = (pathlib.Path(__file__).parent / "..").resolve()  # pylint: disable=no-member
    MODULE_ROOT = PROJECT_ROOT / "reading_comprehension"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "fixtures"

    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / "aug_qanet" / "experiment.json",
                          self.FIXTURES_ROOT / "aug_qanet" / "drop.json")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
