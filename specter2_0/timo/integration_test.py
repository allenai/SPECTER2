"""
Write integration tests for your model interface code here.

The TestCase class below is supplied a `container`
to each test method. This `container` object is a proxy to the
Dockerized application running your model. It exposes a single method:

```
predict_batch(instances: List[Instance]) -> List[Prediction]
```

To test your code, create `Instance`s and make normal `TestCase`
assertions against the returned `Prediction`s.

e.g.

```
def test_prediction(self, container):
    instances = [Instance(), Instance()]
    predictions = container.predict_batch(instances)

    self.assertEqual(len(instances), len(predictions)

    self.assertEqual(predictions[0].field1, "asdf")
    self.assertGreatEqual(predictions[1].field2, 2.0)
```
"""

import logging
import sys
import unittest

import pydantic.error_wrappers

from .interface import Instance, Prediction, TaskType

try:
    from timo_interface import with_timo_container
except ImportError as e:
    logging.warning("""
    This test can only be run by a TIMO test runner. No tests will run. 
    You may need to add this file to your project's pytest exclusions.
    """)
    sys.exit(0)


@with_timo_container
class TestInterfaceIntegration(unittest.TestCase):
    def test__predictions(self, container):
        instances = [Instance(title="Don't Stop Pretraining", task_type=TaskType.DEFAULT)]
        predictions = container.predict_batch(instances)

        self.assertEqual(len(predictions[0].embedding), 768)

        instances.append(Instance(title="Don't Stop Pretraining",
                                  abstract="Language models pretrained on text from a wide variety of sources form the foundation of today\u2019s NLP.",
                                  task_type=TaskType.DEFAULT))
        predictions = container.predict_batch(instances)
        self.assertEqual(len(predictions), 2)
        self.assertNotEqual(predictions[0].embedding[0], predictions[1].embedding[0])

        instances.append(Instance(title="Don't Stop Pretraining",
                                  abstract="Language models pretrained on text from a wide variety of sources form the foundation of today\u2019s NLP.",
                                  task_type=TaskType.CLASSIFICATION))
        instances.append(Instance(title="Don't Stop Pretraining",
                                  abstract="Language models pretrained on text from a wide variety of sources form the foundation of today\u2019s NLP.",
                                  task_type=TaskType.REGRESSION))
        instances.append(Instance(title="Don't Stop Pretraining",
                                  abstract="Language models pretrained on text from a wide variety of sources form the foundation of today\u2019s NLP.",
                                  task_type=TaskType.REGRESSION))
        instances.append(Instance(title="Don't Stop Pretraining",
                                  abstract="Language models pretrained on text from a wide variety of sources form the foundation of today\u2019s NLP.",
                                  task_type=TaskType.PROXIMITY))
        instances.append(Instance(title="Don't Stop Pretraining",
                                  abstract="Language models pretrained on text from a wide variety of sources form the foundation of today\u2019s NLP.",
                                  task_type=TaskType.ADHOC_QUERY))
        predictions = container.predict_batch(instances)
        self.assertEqual(len(predictions), 7)
        self.assertNotAlmostEqual(predictions[-1].embedding[0], predictions[-2].embedding[0])
        for i in range(7):
            if i !=3 and i!=4:
                self.assertNotAlmostEqual(predictions[3].embedding[10], predictions[i].embedding[10])
            else:
                self.assertAlmostEqual(predictions[3].embedding[10], predictions[i].embedding[10])
        import os
        os.environ["use_fp16"] = "True"

        predictions = container.predict_batch(instances)
        #self.assertAlmostEqual(predictions[3].embedding[2], predictions[4].embedding[2])

    def test_invalid_instance(self, container):
        self.assertRaises(pydantic.error_wrappers.ValidationError, Instance)
        self.assertRaises(pydantic.error_wrappers.ValidationError, Instance, title="Don't Stop Pretraining")
        self.assertRaises(pydantic.error_wrappers.ValidationError, Instance, title="Don't Stop Pretraining",
                          abstract=None)
