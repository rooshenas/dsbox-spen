import typing
import importlib
import logging
import pandas as pd
import numpy as np

# importing d3m stuff
from d3m import exceptions
from d3m.container.pandas import DataFrame
from d3m.container.list import List
from d3m.primitive_interfaces.base import CallResult, MultiCallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata import hyperparams, params, base as metadata_base

from dsbox.spen.primitives import config
from dsbox.spen.core import config as cf, mlp
from dsbox.spen.utils.metrics import f1_score, hamming_loss
from dsbox.spen.utils.datasets import get_layers, get_data_val

Inputs = DataFrame
Outputs = DataFrame

class Params(params.Params):
    _mlp_model: mlp.MLP
    _class_name_to_number: typing.List[int]
    _target_column_name: str
    _features: typing.List[str]
    _index: typing.List[int]

class MLCHyperparams(hyperparams.Hyperparams):
    lr = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-3,
        description='Learning rate used during training (fit).'
    )
    lr_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-6,
        description='Learning rate used during training (fit).'
    )
    l2_penalty = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-2,
        description='Learning rate used during training (fit).'
    )
    dropout_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.1,
        description='Learning rate used during training (fit).'
    )
    dimension = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=2,
        description='Learning rate used during training (fit).'
    )
    pred_layer_size = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=2000,
        description='Learning rate used during training (fit).'       
    )
    pred_layer_type = hyperparams.Hyperparameter[str](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default="relu",
        description='Learning rate used during training (fit).'       
    )
    epochs = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=5,
        description='Learning rate used during training (fit).'       
    )
    batch_size = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=100,
        description='Learning rate used during training (fit).'       
    )


class MultiLabelClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, MLCHyperparams]):
    """
    Multi-label classfier primitive
    """

    __author__ = 'UMASS/Pedram Rooshenas'
    metadata = hyperparams.base.PrimitiveMetadata({
        'id': '2dfa8611-a55d-47d6-afb6-e5d531cf5281',
        'version': config.VERSION,
        'name': "dsbox-spen-mlclassifier",
        'description': 'Multi-label classification using SPEN',
        'python_path': 'd3m.primitives.classification.multilabel_classifier.DSBOX',
        'primitive_family': "CLASSIFICATION",
        'algorithm_types': ["DEEP_NEURAL_NETWORK" ],
        'keywords': ['multi-label', 'classification'],
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
        self._fitted = False
        self._inited = False
        self._model = None
        self._training_inputs = None
        self._training_outputs = None
        self._epochs = self.hyperparams["epochs"]
        self._batch_size = self.hyperparams["batch_size"]
        self._config = cf.Config()
        self._config.learning_rate=float(self.hyperparams["lr"])
        self._config.weight_decay=float(self.hyperparams["lr_decay"])
        self._config.dropout=float(self.hyperparams["dropout_rate"])
        self._config.dimension=int(self.hyperparams["dimension"])
        self._config.pred_layer_info=[(self.hyperparams["pred_layer_size"],
                             self.hyperparams["pred_layer_type"])]
    def get_params(self) -> Params:
        param = Params(
                        _mlp_model = self._model,
                        _class_name_to_number = self._labels,
                        _target_column_name = self._label_name,
                        _features = self._features,
                        _index=self._index
                      )
        return param

    def set_params(self, *, params: Params) -> None:
        self._model = params["_mlp_model"]
        self._labels = params["_class_name_to_number"]
        self._label_name = params["_target_column_name"]
        self._features = params["_features"]
        self._index = params["_index"]


    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        if len(inputs) != len(outputs):
            raise ValueError('Training data sequences "inputs" and "outputs" should have the same length.')
        # self._training_size = len(inputs)
        # self._training_inputs = inputs.values
        if 'd3m_index' in outputs.columns:
            self._index = outputs['d3m_index'].tolist()
        outputs=outputs.drop(columns=["d3m_index"])
        self._label_name = outputs.columns[0]
        self._features = list(inputs.columns)
        inputs = inputs.values
        outputs = outputs.values.ravel()
        self._labels = list(self._get_labels(outputs)) # classes
        outputs = self._bit_mapper(outputs, self._labels)
        self._training_inputs, self._val_inputs = inputs[:int(len(inputs)*0.8)], inputs[int(len(inputs)*0.8):]
        self._training_outputs, self._val_outputs = outputs[:int(len(inputs)*0.8)], outputs[int(len(inputs)*0.8):]
        self._config.input_num = self._training_inputs.shape[1]
        self._config.output_num = self._training_outputs.shape[1]


    def fit(self, *, timeout: float = None, iterations: int = None) -> CallResult[None]:
        if self._training_inputs is None:
            raise ValueError('Missing training(fitting) data.')

        if timeout is None:
            timeout = np.inf
        if iterations is None:
            iterations = self._epochs
        if not self._model:
            self._model = self._lazy_init()
        ntrain = len(self._training_inputs)
        loss = []
        for i in range(1, self._epochs):
            bs = min(self._batch_size, ntrain)
            perm = np.random.permutation(ntrain)
            for b in range(ntrain // bs):
                indices = perm[b * bs:(b + 1) * bs]
                xbatch = self._training_inputs[indices][:]
                ybatch = self._training_outputs[indices][:]
                noisex = np.random.normal(xbatch,
                                          np.random.uniform(0,0.5,
                                          np.shape(xbatch)[1])*np.std(xbatch,axis=0),
                                          size=np.shape(xbatch))
                self._model.set_train_iter(i*(ntrain//bs)+b)
                o = self._model.train_batch(noisex, ybatch, verbose=0)
                loss.append(o/bs)
            yval_output = self._model.map_predict(xinput=self._val_inputs)
            ytr_output = self._model.map_predict(xinput=self._training_inputs)
        self._fitted = True
        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        print("start producing")
        if not self._model:
            raise ValueError("model is not fitted or loaded")
        res = self._model.map_predict(xinput=inputs)
        res_df = DataFrame(data=np.array([self._index,
                                            self._genearte_outputs(res)]).T,
                           columns=['d3m_index', self._label_name])
        self._has_finished = True
        self._iterations_done = True
        print("finished")
        return CallResult(res_df, self._has_finished, self._iterations_done)
    

    def _genearte_outputs(self, res): # need improve
        output = []
        labels = list(self._labels)
        for i in range(len(res)):
            tmp = []
            for j, v in enumerate(list(res[i,:])):
                if v == 1.0:
                    tmp.append(str(labels[j]))
            val = ",".join(tmp)
            output.append(val)
        return output 


    def _get_labels(self, target):
        label_set = set()
        for v in target:
            if v.startswith("["):
                for word in v[1,-1].split(","):
                    label_set.add(int(word))
            else:
                for word in v.split(","):
                    label_set.add(int(word))
        return label_set
    

    def _bit_mapper(self, target, label_set):
        columns_list = sorted(list(label_set))
        res_target = np.zeros((len(target), len(columns_list)))
        for i, v in enumerate(target.tolist()):
            if v.startswith("["):
                for word in v.split(","):
                    j = columns_list.index(int(word))
                    res_target[i, j] = 1
            else:
                for word in v.split(","):
                    j = columns_list.index(int(word))
                    res_target[i, j] = 1
        return res_target

    def _lazy_init(self):
        global tf
        tf = importlib.import_module("tensorflow")
        tf.reset_default_graph()
        print(self._config.input_num, self._config.output_num)
        m = mlp.MLP(self._config)
        m.createOptimizer()
        m.get_loss = m.ce_loss
        m.get_prediction_network = m.mlp_prediction_network
        m.construct()
        m.print_vars()
        m.init()
        return m