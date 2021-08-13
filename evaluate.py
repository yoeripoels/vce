"""Testing of explanation methods
"""

from model.r_model import DVAE, VAECE, LVAE, GVAE, ADA_GVAE
from model.cd_model import CD_DVAE
from testing.metric import MetricComputation
import os
import pickle
import argparse
import json

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--create', type=int, default=0)
    parser.add_argument('--metric_name', type=str, default='metric')
    parser.add_argument('--type', type=str, default='')
    parser.add_argument('--load_name', type=str, default='')
    parser.add_argument('--cd_name', type=str, default='cd')
    args = parser.parse_args()

    # initialize our data
    create_metric = args.create == 1
    model_dir = 'trained'
    model_name = os.path.join(model_dir, args.load_name)
    cd_name = os.path.join(model_dir, args.cd_name)
    if create_metric:
        data_base = os.path.join('data', 'synthetic', 'out', 'test')
        x = os.path.join(data_base, 'x')
        y = os.path.join(data_base, 'y')
        y_feature = [os.path.join(data_base, 'y_f{}'.format(i)) for i in range(8)]
        data_lines = pickle.load(open(os.path.join(data_base, 'lines.pkl'), 'rb'))
        classes_0_8 = pickle.load(open(os.path.join(data_base, 'classes.pkl'), 'rb'))
        classes_9 = pickle.load(open(os.path.join(data_base, 'class_9.pkl'), 'rb'))
        data_classes = classes_0_8 + [classes_9[0]]  # only take first variant of 9, so we have 10 items in data_classes

        # create metric object
        mc = MetricComputation(x=x, y=y, y_feature=y_feature, data_lines=data_lines, data_classes=data_classes,
                               num_chunk=5,
                               num_elbo=10000, num_mig=10000, num_acc=10000, num_lacc=10000, num_eac=90,
                               eac_nostyle_ratio=.5)
        mc.save(os.path.join('metrics', args.metric_name))
    else:
        mc = MetricComputation.from_disk(os.path.join('metrics', args.metric_name))
    metric_out = None
    if args.type == 'vaece':
        vaece = VAECE.from_disk(model_name)
        cd = CD_DVAE.from_disk(cd_name)
        vaece.set_cd(cd)
        metric_out = mc.compute_all(vaece)
    elif args.type == 'dvae':
        dvae = DVAE.from_disk(model_name)
        metric_out = mc.compute_all(dvae)
    elif args.type == 'lvae':
        lvae = LVAE.from_disk(model_name)
        metric_out = mc.compute_all(lvae)
    elif args.type == 'gvae':
        gvae = GVAE.from_disk(model_name)
        metric_out = mc.compute_all(gvae)
    elif args.type == 'adagvae':
        adagvae = ADA_GVAE.from_disk(model_name)
        metric_out = mc.compute_all(adagvae)

    # save computed metrics
    if metric_out is not None:
        with open(os.path.join('metrics', args.load_name + '.json'), 'w') as f:
            json.dump(metric_out, f, indent=4)
