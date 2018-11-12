#pylint: disable=unused-import
import pathlib
from allennlp.common.testing import ModelTestCase
from reading_comprehension.qanet import QaNet
from reading_comprehension.qanet_encoder import QaNetEncoder
from reading_comprehension.squad_reader import SquadReader
from reading_comprehension.ema_trainer import EMATrainer


class QANetModelTest(ModelTestCase):

    PROJECT_ROOT = (pathlib.Path(__file__).parent / "..").resolve()  # pylint: disable=no-member
    MODULE_ROOT = PROJECT_ROOT / "reading_comprehension"
    TESTS_ROOT = PROJECT_ROOT / "tests"
    FIXTURES_ROOT = PROJECT_ROOT / "fixtures"

    def setUp(self):
        super().setUp()
        self.set_up_model(self.FIXTURES_ROOT / "qanet" / "experiment.json",
                          self.FIXTURES_ROOT / "qanet" / "squad.json")

    def test_model_can_train_save_and_load(self):
        self.ensure_model_can_train_save_and_load(self.param_file)
