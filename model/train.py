from model.r_model import DVAE, VAECE
from model.cd_model import CD_DVAE

from data import data
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='dvae')
    args = parser.parse_args()
    save_dir = os.path.join('..', 'pretrained')
    if args.type == 'dvae':
        dvae = DVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=2, w_class=10)
        dvae.compile()
        generator, spe = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'), ['x', 'y'])
        dvae.fit(generator, epochs=1, steps_per_epoch=10)
        dvae.save('dvae-test')
        new_m = DVAE.from_disk('dvae-test')
    elif args.type == 'cd':
        cd_dvae = CD_DVAE(input_shape=(32, 32, 1), dim_y=16, dim_x=16, w_kl_x=0.5, w_class=16, w_disc=50)
        cd_dvae.compile()
        generator, spe = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x_pair_full_a', 'x_pair_full_b', 'y_pair', 'x', 'y'])
        cd_dvae.fit(generator, epochs=1, steps_per_epoch=10)
        cd_dvae.save('cd-test')
        new_m = CD_DVAE.from_disk('cd-test')
    elif args.type == 'vaece':
        model_cd = CD_DVAE.from_disk('cd-test')
        vaece = VAECE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=2, w_class=7, w_chg_disc=3, model_cd=model_cd)
        vaece.compile(batch_size=128)
        generator, spe = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x', 'x_p', 'y', 'y_p', 'x_pair_full_a'])
        vaece.fit(generator, epochs=1, steps_per_epoch=10)
        vaece.save('vaece-test')
        new_m = VAECE.from_disk('vaece-test')
    elif args.type == 'adagvae':
        generator, spe = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x_pos_pair_a', 'x_pos_pair_b', 'x', 'y'])
    elif args.type == 'gvae':
        generator, spe = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x_fpair_a', 'x_fpair_b', 'y_fpair', 'x', 'y'])
    elif args.type == 'lvae':
        generator, spe = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x', 'y'] + ['y_f{}'.format(i) for i in range(8)])
