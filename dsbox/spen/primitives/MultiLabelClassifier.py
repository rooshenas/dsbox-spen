import typing
import importlib
import logging

# importing d3m stuff
from d3m import exceptions
from d3m.container.pandas import DataFrame
from d3m.container.list import List
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase

from dsbox.spen.primitives import config

Inputs = DataFrame
Outputs = DataFrame

class Params(params.Params):
    target_column_name: str
    class_name_to_number: typing.Dict[str, int]
    keras_model: Sequential
    feature_shape: typing.List[int]
    input_feature_column_name: str


class NetworkConfig(object):
    def __init__(self, lr, lr_decay, l2_penalty, dropout,
                 dimension, pred_layer_size, pred_layer_type):
        self.lr = lr
        self.lr_decay = lr_decay
        self.l2_penalty = l2_penalty
        self.dropout = dropout
        self.dimension = dimension
        self.pred_layer_size = pred_layer_size
        self.pred_layer_type = pred_layer_type        


class Params(params.Params):
    state: dict


class MLCHyperparams(hyperparams.Hyperparams):
    lr = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-5,
        description='Learning rate used during training (fit).'
    )
    lr_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-5,
        description='Learning rate used during training (fit).'
    )
    l2_penalty = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-5,
        description='Learning rate used during training (fit).'
    )
    dropout_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.5,
        description='Learning rate used during training (fit).'
    )
    dimension = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-5,
        description='Learning rate used during training (fit).'
    )
    pred_layer_size = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-5,
        description='Learning rate used during training (fit).'       
    )
    pred_layer_type = hyperparams.Hyperparameter[str](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-5,
        description='Learning rate used during training (fit).'       
    )


class MultiLabelClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, MLCHyperparams]):
    """
    Multi-label classfier primitive
    """

    __author__ = 'UMASS/Pedram Rooshenas'
    metadata = metadata.PrimitiveMetadata({
        'id': '2dfa8611-a55d-47d6-afb6-e5d531cf5281',
        'version': config.VERSION,
        'name': "dsbox-spen-mlclassifier",
        'description': 'Multi-label classification using SPEN',
        'python_path': 'd3m.primitives.dsbox.multilabel_classifier',
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


    def __init__(self, *, hyperparams: MLCHyperparams) -> None:
        super().__init__(hyperparams=hyperparams)
        self.hyperparams = hyperparams
        self._has_finished = False
        self._iterations_done = 0
        self._inited = False
        self._model = None
        self._training_inputs = None
        self._training_outputs = None
        self._config = NetworkConfig(
            lr=hyperparams["lr"],
            lr_decay=hyperparams["lr_decay"],
            dropout=hyperparams["dropout_rate"],
            dimension=hyperparams["dimension"],
            pred_layer_size=hyperparams["pred_layer_size"],
            pred_layer_type=hyperparams["pred_layer_type"]
        )


    def get_params(self) -> Params:
        param = Params(
                        keras_model = self._model,
                        class_name_to_number = self._class_name_to_number,
                        target_column_name = self._target_column_name,
                        feature_shape = self._feature_shape,
                        input_feature_column_name = self._input_feature_column_name
                      )
        return param

    def set_params(self, *, params: Params) -> None:
        self._model = params["keras_model"]
        self._class_name_to_number = params["class_name_to_number"]
        self._target_column_name = params["target_column_name"]
        self._feature_shape = params["feature_shape"]
        self._input_feature_column_name = params["input_feature_column_name"]


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        if len(inputs) != len(outputs):
            raise ValueError('Training data sequences "inputs" and "outputs" should have the same length.')
        self._training_size = len(inputs)
        self._training_inputs = inputs
        self._training_outputs = outputs

        self._fitted = False


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        pass


