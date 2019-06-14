import copy
import enum
import importlib
import random
import typing

import numpy as np
import pandas as pd

# importing d3m stuff
# from d3m import exceptions
from d3m.container.pandas import DataFrame
from d3m.primitive_interfaces.base import CallResult
from d3m.primitive_interfaces.supervised_learning import SupervisedLearnerPrimitiveBase
from d3m.metadata import hyperparams, params

from dsbox.spen.primitives import config
from dsbox.spen.core import config as cf, sg_spen
# from dsbox.spen.utils.metrics import f1_score, hamming_loss
# from dsbox.spen.utils.datasets import get_layers, get_data_val
import types


Inputs = DataFrame
Outputs = DataFrame


class LabelStyle(enum.Enum):
    CSV = 1
    ARRAY = 2
    ONE_LABEL_PER_ROW = 3
    SINGLE_LABEL = 4

    # CSV example:
    # 1, '3,25,34'

    # Array example:
    # 1, np.array(['3', '25', '34'])

    # ONE_LABEL_PER_ROW example:
    # 1, '3'
    # 1, '25'
    # 1, '34'


class Params(params.Params):
    _mlp_model: sg_spen.SPEN
    _class_name_to_number: typing.List[str]
    _target_column_name: str
    _features: typing.List[str]
    _index: typing.List[str]


class MLCHyperparams(hyperparams.Hyperparams):
    lr = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-3,
        description='Learning rate used during training (fit).'
    )
    lr_decay = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-6,
        description='Learning rate decay.'
    )
    l2_penalty = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=1e-2,
        description='L2 penalty'
    )
    dropout_rate = hyperparams.Hyperparameter[float](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=0.1,
        description='Dropout rate'
    )
    dimension = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=2,
        description='Dimension'
    )
    pred_layer_size = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=2000,
        description='Prediction layer size'
    )
    pred_layer_type = hyperparams.Hyperparameter[str](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default="relu",
        description='Prediction layer type'
    )
    epochs = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=5,
        description='Epochs'
    )
    batch_size = hyperparams.Hyperparameter[int](
        semantic_types=['https://metadata.datadrivendiscovery.org/types/ControlParameter'],
        default=100,
        description='Batch size'
    )


class MLClassifier(SupervisedLearnerPrimitiveBase[Inputs, Outputs, Params, MLCHyperparams]):
    """
    Multi-label classifier primitive using structured prediction energy networks (SPEN).
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
            'contact': config.D3M_CONTACT,
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
        self._label_style = LabelStyle.CSV
        self._index = []
        self._training_inputs = None
        self._training_outputs = None
        self._epochs = self.hyperparams["epochs"]
        self._batch_size = self.hyperparams["batch_size"]
        self._config = cf.Config()
        self._config.learning_rate = self.hyperparams["lr"]
        self._config.weight_decay = self.hyperparams["lr_decay"]
        self._config.dropout = self.hyperparams["dropout_rate"]
        self._config.dimension = self.hyperparams["dimension"]
        self._config.pred_layer_info = [(self.hyperparams["pred_layer_size"],
                                         self.hyperparams["pred_layer_type"])]
        self._config.use_search = False
        self._config.en_layer_info = [(15, 'softplus')]
        self._config.layer_info = [(1000, 'relu')]
        self._config.noise_rate = 2*self.hyperparams["lr"]
        self._config.alpha = 1.0

        self._config.l2_penalty = self.hyperparams["l2_penalty"]
        self._config.inf_iter = 10
        self._config.inf_rate = 0.5
        self._config.margin_weight = 100.0
        # self._config.output_num = output_num
        # config.input_num = input_num
        self._config.inf_penalty = 0.01
        self._config.loglevel = 0
        self._config.score_margin = 0.01
        self._config.score_max = 1.0

    def get_params(self) -> Params:
        param = Params(
            _mlp_model=self._model,
            _class_name_to_number=self._labels,
            _target_column_name=self._label_name,
            _features=self._features,
            _index=self._index,
            _label_style=self._label_style,
        )
        return param

    def set_params(self, *, params: Params) -> None:
        self._model = params["_mlp_model"]
        self._labels = params["_class_name_to_number"]
        self._label_name = params["_target_column_name"]
        self._features = params["_features"]
        self._index = params["_index"]
        self._label_style = params["_label_style"]

    def set_training_data(self, *, inputs: Inputs, outputs: Outputs) -> None:
        if len(inputs) == 0:
            raise ValueError('Training data size is zero')

        if len(inputs) != len(outputs):
            raise ValueError('Training data sequences "inputs" and "outputs" should have the same length.')

        self._label_style = self._detect_label_style(inputs, outputs)

        if 'd3mIndex' in outputs.columns:
            outputs = outputs.drop(columns=["d3mIndex"])

        if self._label_style == LabelStyle.ONE_LABEL_PER_ROW:
            # convert to array style
            inputs, outputs = self._convert_to_array_style(inputs, outputs)

        self._label_name = outputs.columns[0]
        if 'd3mIndex' in inputs.columns:
            self._index = inputs['d3mIndex'].tolist()
            inputs = inputs.drop(columns=["d3mIndex"])
        self._features = list(inputs.columns)

        inputs = inputs.values
        outputs = outputs.values.ravel()

        if self._label_style == LabelStyle.CSV:
            # Comma separated labels
            self._labels = sorted(list(self._get_csv_labels(outputs)))
            outputs = self._bit_mapper_csv(outputs, self._labels)
        elif self._label_style == LabelStyle.ONE_LABEL_PER_ROW or self._label_style == LabelStyle.ARRAY:
            # ONE_LABEL_PER_ROW converted to ARRAY label
            self._labels = sorted(list(self._get_array_labels(outputs)))
            outputs = self._bit_mapper_array(outputs, self._labels)
        elif self._label_style == LabelStyle.SINGLE_LABEL:
            self._labels = sorted(list(self._get_labels(outputs)))
            outputs = self._bit_mapper(outputs, self._labels)
        else:
            raise ValueError(f'Label encoding style not recognized: {self._label_style}')
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
                                          np.random.uniform(0, 0.5,
                                                            np.shape(xbatch)[1])*np.std(xbatch, axis=0),
                                          size=np.shape(xbatch))
                self._model.set_train_iter(i*(ntrain//bs)+b)
                o = self._model.train_batch(noisex, ybatch, verbose=0)
                # loss.append(o/bs)
            yval_output = self._model.map_predict(xinput=self._val_inputs)
            ytr_output = self._model.map_predict(xinput=self._training_inputs)
        self._fitted = True
        return CallResult(None)

    def produce(self, *, inputs: Inputs, timeout: float = None, iterations: int = None) -> CallResult[Outputs]:
        print("start producing")
        if not self._model:
            raise ValueError("model is not fitted or loaded")

        index = None
        if 'd3mIndex' in inputs.columns:
            index = inputs['d3mIndex']
            inputs = inputs.drop(columns=["d3mIndex"])
        if self._label_style == LabelStyle.ONE_LABEL_PER_ROW:
            inputs_nd = self._unique_index(inputs, index).values
        else:
            inputs_nd = inputs.values
        res = self._model.map_predict(xinput=inputs_nd)

        if index is not None:
            if self._label_style == LabelStyle.CSV:
                res_df = DataFrame(data=np.array([index,
                                                  self._generate_outputs_csv(res)]).T,
                                   columns=['d3mIndex', self._label_name])
            elif self._label_style == LabelStyle.ONE_LABEL_PER_ROW:
                res_df = DataFrame(data=np.array([index,
                                                  self._generate_outputs_single(res, index)]).T,
                                   columns=['d3mIndex', self._label_name])
            elif self._label_style == LabelStyle.ARRAY:
                res_df = DataFrame(data=np.array([index,
                                                  self._generate_outputs_array(res)]).T,
                                   columns=['d3mIndex', self._label_name])
            elif self._label_style == LabelStyle.SINGLE_LABEL:
                res_df = DataFrame(data=np.array([index,
                                                  self._generate_outputs(res)]).T,
                                   columns=['d3mIndex', self._label_name])

        else:
            res_df = DataFrame(data=np.array(self._generate_outputs(res)).T, columns=[self._label_name])
        self._has_finished = True
        self._iterations_done = True
        print("finished")
        return CallResult(res_df, self._has_finished, self._iterations_done)

    def _generate_outputs(self, res):
        output = []
        labels = self._labels
        for i in range(len(res)):
            for j, v in enumerate(list(res[i, :])):
                if v == 1.0:
                    output.append(labels[j])
                    break
                if j == len(list(res[i, :]))-1:
                    output.append((labels[-1]))
                    break
        return output

    def _generate_outputs_csv(self, res): # need improve
        output = []
        labels = self._labels

        for i in range(len(res)):
            tmp = []
            for j, v in enumerate(list(res[i, :])):
                if v == 1.0:
                    tmp.append(str(labels[j]))
            val = ",".join(tmp)
            output.append(val)
        return output

    def _generate_outputs_array(self, res): # need improve
        output = []
        labels = self._labels

        for i in range(len(res)):
            tmp = []
            for j, v in enumerate(list(res[i, :])):
                if v == 1.0:
                    tmp.append(str(labels[j]))
            output.append(np.asarray(tmp))
        return output

    def _generate_outputs_single(self, res, index):
        output = []
        labels = self._labels

        # Count number of repeats for each index value
        repeats = {}
        for _, d3mIndex in index.iteritems():
            if d3mIndex in repeats:
                repeats[d3mIndex] = repeats[d3mIndex] + 1
            else:
                repeats[d3mIndex] = 1

        unique_index = index.unique()
        output_dict = {}
        for d3mIndex, repeat in repeats.items():
            i = np.where(unique_index == d3mIndex)[0][0]
            label_list = []
            count = 0
            for j, v in enumerate(list(res[i, :])):
                if v == 1.0:
                    label_list.append(str(labels[j]))
                    count += 1
                    if count == repeat:
                        break
            if (count < repeat):
                # Not enough labels, select rest randomly
                for label in random.choices(labels, k=repeat-count):
                    label_list.append(str(label))
            output_dict[d3mIndex] = label_list

        for _, d3mIndex in index.iteritems():
            output.append(output_dict[d3mIndex].pop())
        return output

    def _detect_label_style(self, inputs, outputs) -> LabelStyle:
        if 'd3mIndex' in inputs:
            index = inputs['d3mIndex']
            if (len(set(index))) < len(index):
                return LabelStyle.ONE_LABEL_PER_ROW

        sample = outputs.iloc[0, -1]
        if isinstance(sample, np.ndarray) or isinstance(sample, pd.Series):
            return LabelStyle.ARRAY

        commas = 0
        for value in outputs.iloc[:, -1]:
            commas += value.count(',')

        if commas > 0:
            return LabelStyle.CSV
        else:
            return LabelStyle.SINGLE_LABEL

    def _convert_to_array_style(self, inputs, outputs):
        ''''
        Convert single label per row format to array of labels format
        '''
        index_column = inputs['d3mIndex']

        new_inputs = inputs.iloc[:len(index_column), :]
        new_outputs = outputs.iloc[:len(index_column), :]

        labels = {}
        for row_index, value in index_column.iteritems():
            label = outputs.iloc[row_index, 0]
            if value in labels:
                labels[value].append(label)
            else:
                new_inputs.iloc[len(labels), 0] = new_inputs.iloc[row_index, 0]
                labels[value] = [label]
        for row_index, value in new_inputs['d3mIndex'].iteritems():
            new_outputs.iloc[row_index, 0] = labels[value]
        return new_inputs, new_outputs

    def _unique_index(self, inputs: DataFrame, index: pd.Series) -> DataFrame:
        '''
        Remove rows with duplicate index
        '''
        unique_index = index.unique()
        new_inputs = pd.DataFrame(inputs.iloc[:len(unique_index), :])
        seen = set()
        for row_index, row in inputs.iterrows():
            if index[row_index] not in seen:
                new_inputs.iloc[len(seen), :] = row
                seen.add(index[row_index])
        return new_inputs

    def _get_array_labels(self, target):
        label_set = set()
        for array in target:
            for v in array:
                label_set.add(v)
        return label_set

    def _get_labels(self, target):
        label_set = set()
        for v in target:
            label_set.add(v)
        return label_set

    def _get_csv_labels(self, target):
        label_set = set()
        for v in target:
            if v.startswith("["):
                for word in v[1, -1].split(","):
                    label_set.add(word)
            else:
                for word in v.split(","):
                    label_set.add(word)
        return label_set

    def _bit_mapper(self, target, label_list):
        columns_list = label_list
        res_target = np.zeros((len(target), len(columns_list)))
        target_copy = copy.copy(target)
        for i in range(target_copy.shape[0]):
            j = columns_list.index(target_copy[i])
            res_target[i, j] = 1
        return res_target

    def _bit_mapper_csv(self, target, label_list):
        columns_list = label_list
        res_target = np.zeros((len(target), len(columns_list)))
        for i, v in enumerate(target.tolist()):
            if v.startswith("["):
                for word in v.split(","):
                    j = columns_list.index(word)
                    res_target[i, j] = 1
            else:
                for word in v.split(","):
                    j = columns_list.index(word)
                    res_target[i, j] = 1
        return res_target

    def _bit_mapper_array(self, target, label_list):
        columns_list = label_list
        res_target = np.zeros((len(target), len(columns_list)))
        for i, v in enumerate(target.tolist()):
            for word in v:
                j = columns_list.index(word)
                res_target[i, j] = 1
        return res_target

    def _lazy_init(self):
        global tf, tflearn, tfi

        tflearn = importlib.import_module('tflearn')
        tfi = importlib.import_module('tflearn.initializations')
        tf = importlib.import_module("tensorflow")
        tf.reset_default_graph()
        # print(self._config.input_num, self._config.output_num)
        # m = mlp.MLP(self._config)
        # m.createOptimizer()
        # m.get_loss = m.ce_loss
        # m.get_prediction_network = m.mlp_prediction_network
        # m.construct()
        # m.init()
        # return m

        s = sg_spen.SPEN(self._config)
        s.createOptimizer()
        s.evaluate = evaluate_score
        s.get_energy = types.MethodType(get_energy_mlp, s)
        s.train_batch = s.train_unsupervised_sg_batch
        s.construct(training_type=sg_spen.TrainingType.Rank_Based)
        print("Energy:")
        # s.print_vars()
        s.init()
        return s


def get_energy_mlp(self, xinput=None, yinput=None, embedding=None, reuse=False):
    output_size = yinput.get_shape().as_list()[-1]
    with tf.variable_scope(self.config.spen_variable_scope):
        with tf.variable_scope(self.config.fx_variable_scope) as scope:
            net = xinput
            j = 0
            for (sz, a) in self.config.layer_info:
                net = tflearn.fully_connected(net, sz,
                                              weight_decay=self.config.weight_decay, activation=a,
                                              weights_init=tfi.variance_scaling(),
                                              bias_init=tfi.zeros(), regularizer='L2', reuse=reuse,
                                              scope=("fx.h" + str(j)))
                net = tflearn.dropout(net, 1.0 - self.config.dropout)
                j = j + 1
            logits = tflearn.fully_connected(net, output_size, activation='linear', regularizer='L2',
                                             weight_decay=self.config.weight_decay,
                                             weights_init=tfi.variance_scaling(), bias_init=tfi.zeros(),
                                             reuse=reuse, scope="fx.fc")

            mult = logits * yinput
            local_e = tf.reduce_sum(mult, axis=1)
        with tf.variable_scope(self.config.en_variable_scope) as scope:
            j = 0
            net = yinput
            for (sz, a) in self.config.en_layer_info:
                net = tflearn.fully_connected(net, sz,
                                              weight_decay=self.config.weight_decay,
                                              weights_init=tfi.variance_scaling(),
                                              activation=a,
                                              bias=False,
                                              reuse=reuse, regularizer='L2',
                                              scope=("en.h" + str(j)))

                j = j + 1
            global_e = tf.squeeze(
                tflearn.fully_connected(net, 1, activation='linear', weight_decay=self.config.weight_decay,
                                        weights_init=tfi.variance_scaling(), bias=False,
                                        reuse=reuse, regularizer='L2',
                                        scope=("en.g")))

    return tf.squeeze(tf.add(local_e, global_e))


def f1_score_c_ar(cpred, ctrue):
    intersection = np.sum(np.minimum(cpred, ctrue), 1)
    union = np.sum(np.maximum(cpred, ctrue), 1)
    return np.divide(2.0 * intersection, union + intersection)


def evaluate_score(xinput=None, yinput=None, yt=None):
    return np.array(f1_score_c_ar(yinput, yt))
