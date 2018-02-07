from d3m_metadata.container.numpy import ndarray
from d3m_metadata import hyperparams, params, metadata, utils
from primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from primitive_interfaces.base import CallResult

from typing import Dict, List, Tuple, Type

from dsbox.spen.primitives import config

Inputs = ndarray
Outputs = ndarray



class Params(params.Params):
    state: Dict


class Hyperparams(hyperparams.Hyperparams):
  pass

class MultiLabelClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, Hyperparams]):
  """
  Multi-label classfier primitive
  """

  __author__ = 'UMASS/Pedram Rooshenas'
  metadata = metadata.PrimitiveMetadata({
    'id': '2dfa8611-a55d-47d6-afb6-e5d531cf5281',
    'version': config.VERSION,
    'name': "dsbox-spen-mlclassifier",
    'description': 'Multi-label classification using SPEN',
    'python_path': 'd3m.primitives.dsbox.MLCLassifier',
    'primitive_family': metadata.PrimitiveFamily.SupervisedClassification,
    'algorithm_types': [metadata.PrimitiveAlgorithmType.FEEDFORWARD_NEURAL_NETWORK, ],
    'keywords': ['spen', 'multi-label', 'classification'],
    'source': {
      'name': config.D3M_PERFORMER_TEAM,
      'uris': [config.REPOSITORY]
    },
    # The same path the primitive is registered with entry points in setup.py.
    'installation': [config.INSTALLATION],
    # Choose these from a controlled vocabulary in the schema. If anything is missing which would
    # best describe the primitive, make a merge request.

    # A metafeature about preconditions required for this primitive to operate well.
    'precondition': [],
    'hyperparms_to_tune': []
  })

  def __init__(self):
    pass



  def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
    if len(inputs) != len(outputs):
      raise ValueError('Training data sequences "inputs" and "outputs" should have the same length.')
    self._training_size = len(inputs)
    self._training_inputs = inputs
    self._training_outputs =  outputs

    self._fitted = False


  def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
    pass


