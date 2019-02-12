import sys
from allennlp.predictors import Predictor
from allennlp.models.archival import load_archive
from allennlp.common.util import import_submodules


# The path to the augmented qanet project dir
sys.path.append('../../')
import_submodules('reading_comprehension')


# This maps from the name of the task
# to the ``DemoModel`` indicating the location of the trained model
# and the type of the ``Predictor``.  This is necessary, as you might
# have multiple models (for example, a NER tagger and a POS tagger)
# that have the same ``Predictor`` wrapper. The corresponding model
# will be served at the `/predict/<name-of-task>` API endpoint.

class DemoModel:
    """
    A demo model is determined by both an archive file
    (representing the trained model)
    and a choice of predictor
    """

    def __init__(self, archive_file: str, predictor_name: str) -> None:
        self.archive_file = archive_file
        self.predictor_name = predictor_name

    def predictor(self) -> Predictor:
        archive = load_archive(self.archive_file)
        return Predictor.from_archive(archive, self.predictor_name)


MODELS = {'machine-comprehension': DemoModel('../../model.tar.gz',  # the path to the model archive file
                                             'machine-comprehension')}
