"""Metric-computation class: insert data and parameters. Then evaluate all models with same data.
"""
import numpy as np
from data.synthetic.structure import ShapeParser
from data.data import get_var_info
from expl.evaluation import create_explanation_simple
import random
import os
import pickle


def parse_split_data(dir_name, type_name, num_chunks):
    num_chunk, total_elem = get_var_info(dir_name)
    if type_name in num_chunks:
        num_chunk = num_chunks[type_name]
    shape = np.load(os.path.join(dir_name, '0.npy')).shape
    n_c, chunk_shape = shape[0], shape[1:]
    data_return = np.zeros((n_c * num_chunk, *chunk_shape))
    for i in range(num_chunk):
        data_return[i * n_c:(i+1) * n_c] = np.load(os.path.join(dir_name, '{}.npy'.format(i)))
    return data_return


class MetricComputation:
    def __init__(self, x=None, y=None, y_feature=None, data_lines=None, data_classes=None,
                 num_chunks=None,
                 num_elbo=1000, num_mig=1000, num_acc=1000, num_lacc=1000, num_eac=50, eac_nostyle_ratio=.5):
        # data sets / data generators
        if num_chunks is None:
            num_chunks = {}

        if isinstance(x, str):
            self.x = parse_split_data(x, 'x', num_chunks)
        elif isinstance(x, np.ndarray):
            self.x = x
        else:
            self.x = None

        if isinstance(y, str):
            self.y = parse_split_data(y, 'y', num_chunks)
        elif isinstance(y, np.ndarray):
            self.y = y
        else:
            self.y = None

        if isinstance(y_feature, list) and isinstance(y_feature[0], str):
            y_feature = [parse_split_data(y, 'y_f', num_chunks) for y in y_feature]  # process chunk-dirs to arrays
        if isinstance(y_feature, list) and isinstance(y_feature[0], np.ndarray):
            # process separate feature arrays to single ndarray
            n = y_feature[0].shape[0]
            num_feature = len(y_feature)
            self.y_feature = np.zeros((n, num_feature))
            for i in range(num_feature):
                self.y_feature[:, [i]] = y_feature[:, [1]]
        elif isinstance(y_feature, np.ndarray):
            self.y_feature = y_feature
        else:
            self.y_feature = None

        # data source, for explanation evaluation
        self.data_lines = data_lines
        self.data_classes = data_classes

        # check which metrics we can compute based on supplied data
        self.config = {'elbo': self.x is not None, 'acc': self.y is not None, 'mig': self.y_feature is not None,
                       'eac': self.data_lines is not None and self.data_classes is not None}

        # parameters
        self.num_elbo = num_elbo
        self.num_mig = num_mig
        self.num_acc = num_acc
        self.num_lacc = num_lacc
        self.num_eac = num_eac
        self.eac_nostyle_ratio = eac_nostyle_ratio

        # which indices we use, to keep computations consistent
        if self.config['elbo']:
            self.idx_elbo = np.random.randint(low=0, high=x.shape[0], size=self.num_elbo)
        if self.config['acc']:
            self.idx_acc = np.random.randint(low=0, high=x.shape[0], size=self.num_acc)
            self.idx_lacc = np.random.randint(low=0, high=x.shape[0], size=self.num_lacc)
        if self.config['mig']:
            self.idx_mig = np.random.randint(low=0, high=x.shape[0], size=self.num_mig)

        if self.config['eac']:  # pre-generate explanation pairs
            h, w = self.x.shape[1:3]
            self.sp = ShapeParser(w=w, h=h)

            # create all potential pairs
            num_classes = len(data_classes)
            all_combination = []
            for i in range(num_classes):
                for j in range(num_classes):
                    if i != j:
                        all_combination.append((i, j))
            num_repetition = (num_eac - 1) // len(all_combination)
            all_combination = all_combination * (1 + num_repetition)
            random.shuffle(all_combination)
            self.eac_pair = all_combination[:num_eac]
            self.eac_pair_modification = []

            num_before = int(self.num_eac * self.eac_nostyle_ratio)
            for i, (a, b) in enumerate(self.eac_pair):
                # create a sample explanation so we can create a 'random modification' and normalize it
                # with respect to an expected shape
                sample_exp = create_explanation_simple(self.data_lines, self.data_classes[a], self.data_classes[b],
                                                       return_shape=True)
                if i < num_before:  # if we are to share the style / modification
                    mod = self.sp.get_random_modification(sample_exp)
                    self.eac_pair_modification.append((mod, mod))
                else:  # two different modifications
                    self.eac_pair_modification.append((self.sp.get_random_modification(sample_exp),
                                                       self.sp.get_random_modification(sample_exp)))

    def save(self, save_name):
        if self.config['elbo']:
            np.save(save_name + '-x.npy', self.x)
            np.save(save_name + '-idx_elbo.npy', self.idx_elbo)
        if self.config['acc']:
            np.save(save_name + '-y.npy', self.y)
            np.save(save_name + '-idx_acc.npy', self.idx_acc)
            np.save(save_name + '-idx_lacc.npy', self.idx_lacc)
        if self.config['mig']:
            np.save(save_name + '-y_feature.npy', self.y_feature)
            np.save(save_name + '-idx_mig.npy', self.idx_mig)
        if self.config['eac']:
            pickle.dump(self.data_lines, open(save_name + '-exp-lines.pkl', 'wb'))
            pickle.dump(self.data_classes, open(save_name + '-exp-classes.pkl', 'wb'))
            pickle.dump(self.eac_pair, open(save_name + '-exp-pair.pkl', 'wb'))
            pickle.dump(self.eac_pair_modification, open(save_name + '-exp-pair-mod.pkl', 'wb'))
            pickle.dump((self.num_eac, self.eac_nostyle_ratio), open(save_name + '-exp-setting.pkl', 'wb'))
        pickle.dump(self.config, open(save_name + '-config.pkl', 'wb'))

    @classmethod
    def load(cls, load_name):
        config = pickle.load(open(load_name + '-config.pkl', 'rb'))
        mc = cls()
        if config['elbo']:
            mc.x = np.load(load_name + '-x.npy')
            mc.idx_elbo = np.load(load_name, '-idx_elbo.npy')
            mc.num_elbo = mc.idx_elbo.shape[0]
        if config['acc']:
            mc.y = np.load(load_name + '-y.npy')
            mc.idx_acc = np.load(load_name + '-idx_acc.npy')
            mc.num_acc = mc.idx_acc.shape[0]
            mc.idx_lacc = np.load(load_name + 'idx_lacc.npy')
            mc.num_lacc = mc.idx_lacc.shape[0]
        if config['mig']:
            mc.y_feature = np.load(load_name + '-y_feature.npy')
            mc.idx_mig = np.load(load_name + '-idx_mig.npy')
            mc.num_mig = mc.idx_mig.shape[0]
        if config['eac']:
            mc.data_lines = pickle.load(open(load_name + '-exp-lines.pkl', 'rb'))
            mc.data_classes = pickle.load(open(load_name + '-exp-classes.pkl', 'rb'))
            mc.eac_pair = pickle.load(open(load_name + '-exp-pair.pkl', 'rb'))
            mc.eac_pair_modification = pickle.load(open(load_name + '-exp-pair-mod.pkl', 'rb'))
            mc.num_eac, mc.eac_nostyle_ratio = pickle.load(open(load_name + '-exp-setting.pkl', 'rb'))
        mc.config = config
        return mc

    def mig(self, model):
        if not self.config['mig']:
            return False

    def elbo(self, model):
        if not self.config['elbo']:
            return False

    def acc(self, model):
        if not self.config['acc']:
            return False

    def lacc(self, model):
        if not self.config['acc']:
            return False

    def eac(self, model):
        if not self.config['eac']:
            return False



# mig

# elbo (rec/kl)

# acc from own classifier

# acc using logistic regression classifier

# eac
