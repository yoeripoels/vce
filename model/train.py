"""Training of explanation models.
"""

from model.r_model import DVAE, VAECE, GVAE, LVAE, ADA_GVAE
from model.cd_model import CD_DVAE

from data import data
import os
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, default='dvae')
    args = parser.parse_args()
    save_dir = os.path.join('..', 'pretrained', 'test')
    if args.type == 'dvae':
        dvae = DVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=2, w_class=10)
        generator, spe, batch_size = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'), ['x', 'y'])
        dvae.compile()
        dvae.fit(generator, epochs=1, steps_per_epoch=10)
        dvae.save(os.path.join(save_dir, 'dvae-test'))
        new_m = DVAE.from_disk(os.path.join(save_dir, 'dvae-test'))
    elif args.type == 'cd':
        cd_dvae = CD_DVAE(input_shape=(32, 32, 1), dim_y=16, dim_x=16, w_kl_x=0.5, w_class=16, w_chg_disc=50)
        generator, spe, batch_size = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x_pair_full_a', 'x_pair_full_b', 'y_pair', 'x', 'y'])
        cd_dvae.compile()
        cd_dvae.fit(generator, epochs=1, steps_per_epoch=10)
        cd_dvae.save(os.path.join(save_dir, 'cd-test'))
        new_m = CD_DVAE.from_disk(os.path.join(save_dir, 'cd-test'))
    elif args.type == 'vaece':
        model_cd = CD_DVAE.from_disk(os.path.join(save_dir, 'cd-test'))
        vaece = VAECE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=2, w_class=7, w_chg_disc=3, model_cd=model_cd)
        generator, spe, batch_size = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x', 'x_p', 'y', 'y_p', 'x_pair_full_a'])
        vaece.compile(batch_size=batch_size)
        vaece.fit(generator, epochs=1, steps_per_epoch=10)
        vaece.save(os.path.join(save_dir, 'vaece-test'))
        new_m = VAECE.from_disk(os.path.join(save_dir, 'vaece-test'))
    elif args.type == 'gvae':
        gvae = GVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=1, w_class=6)
        generator, spe, batch_size = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x_fpair_a', 'x_fpair_b', 'y_fpair', 'x', 'y'])
        gvae.compile(batch_size=batch_size)
        gvae.fit(generator, epochs=1, steps_per_epoch=10)
        gvae.save(os.path.join(save_dir, 'gvae-test'))
        new_m = GVAE.from_disk(os.path.join(save_dir, 'gvae-test'))
    elif args.type == 'lvae':
        lvae = LVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=1, w_class=7, w_label=20)
        generator, spe, batch_size = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x', 'y'] + ['y_f{}'.format(i) for i in range(8)])
        lvae.compile(batch_size=batch_size)
        lvae.fit(generator, epochs=1, steps_per_epoch=10)
        lvae.save(os.path.join(save_dir, 'lvae-test'))
        new_m = LVAE.from_disk(os.path.join(save_dir, 'lvae-test'))
    elif args.type == 'adagvae':
        ada_gvae = ADA_GVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=1, w_class=4)
        generator, spe, batch_size = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'),
                                            ['x_pos_pair_a', 'x_pos_pair_b', 'x', 'y'])
        ada_gvae.compile(batch_size=batch_size)
        ada_gvae.fit(generator, epochs=1, steps_per_epoch=10)
        ada_gvae.save(os.path.join(save_dir, 'adagvae-test'))
        new_m = ADA_GVAE.from_disk(os.path.join(save_dir, 'adagvae-test'))
    '''
    Look into using ADA-GVAE with arbitrary grouped images (i.e., not k-1), 
    and GVAE with all dimensions set at once
    '''
