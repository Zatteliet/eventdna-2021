import logging

from experiments import corpus

logger = logging.getLogger(__name__)


def test_featurized_examples():
    examples = corpus.get_examples(main_events_only=True)
    assert len(examples) > 0
    logger.info(examples[0])
