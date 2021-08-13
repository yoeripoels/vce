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
    parser.add_argument('--epoch', type=int, default=10)
    parser.add_argument('--save', type=bool, default=True)
    parser.add_argument('--save_name', type=str, default='')
    parser.add_argument('--load_name', type=str, default='')
    parser.add_argument('--cd_name', type=str, default='cd')
    parser.add_argument('--verbose', type=int, default=2)
    args = parser.parse_args()
    if args.save_name == '':
        save_name = args.type
    else:
        save_name = args.save_name
    verbose = args.verbose
    epochs = args.epoch
    save_dir = os.path.join('trained')
    data_path = os.path.join('data', 'synthetic', 'out')
    if args.type == 'dvae':
        dvae = DVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=2, w_class=10)
        generator, spe, batch_size = data.get_data_disk(data_path, ['x', 'y'])
        dvae.compile()
        dvae.fit(generator, epochs=epochs, steps_per_epoch=spe, verbose=verbose)
        dvae.save(os.path.join(save_dir, save_name))
    elif args.type == 'cd':
        cd_dvae = CD_DVAE(input_shape=(32, 32, 1), dim_y=16, dim_x=16, w_kl_x=0.5, w_class=16, w_chg_disc=50, w_full=2)
        generator, spe, batch_size = data.get_data_disk(data_path,
                                            ['x_pair_full_a', 'x_pair_full_b', 'y_pair', 'x', 'y'])
        cd_dvae.compile()
        cd_dvae.fit(generator, epochs=epochs, steps_per_epoch=spe, verbose=verbose)
        cd_dvae.save(os.path.join(save_dir, save_name))
    elif args.type == 'vaece':
        model_cd = CD_DVAE.from_disk(os.path.join(save_dir, args.cd_name))
        vaece = VAECE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=2, w_class=7, w_chg_disc=3, model_cd=model_cd)
        generator, spe, batch_size = data.get_data_disk(data_path,
                                            ['x', 'x_p', 'y', 'y_p', 'x_pair_full_a'])
        vaece.compile(batch_size=batch_size)
        vaece.fit(generator, epochs=epochs, steps_per_epoch=spe, verbose=verbose)
        vaece.save(os.path.join(save_dir, save_name))
    elif args.type == 'gvae':
        gvae = GVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=1, w_class=6)
        generator, spe, batch_size = data.get_data_disk(data_path,
                                            ['x_fpair_a', 'x_fpair_b', 'y_fpair', 'x', 'y'])
        gvae.compile(batch_size=batch_size)
        gvae.fit(generator, epochs=epochs, steps_per_epoch=spe, verbose=verbose)
        gvae.save(os.path.join(save_dir, save_name))
    elif args.type == 'lvae':
        lvae = LVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=1, w_class=7, w_label=20)
        generator, spe, batch_size = data.get_data_disk(data_path,
                                            ['x', 'y'] + ['y_f{}'.format(i) for i in range(8)], batch_size=128)
        lvae.compile(batch_size=batch_size)
        lvae.fit(generator, epochs=epochs, steps_per_epoch=spe, verbose=verbose)
        lvae.save(os.path.join(save_dir, save_name))
    elif args.type == 'adagvae':
        ada_gvae = ADA_GVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=1, w_class=4)
        generator, spe, batch_size = data.get_data_disk(data_path,
                                            ['x_pos_pair_a', 'x_pos_pair_b', 'x', 'y'])
        ada_gvae.compile(batch_size=batch_size)
        ada_gvae.fit(generator, epochs=epochs, steps_per_epoch=spe, verbose=verbose)
        ada_gvae.save(os.path.join(save_dir, save_name))
