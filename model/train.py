from model.r_model import DVAE

from data import data
import os

if __name__ == '__main__':
    dvae = DVAE(input_shape=(32, 32, 1), dim_y=8, dim_x=8, w_kl_y=2, w_class=10)
    dvae.compile()

    generator, spe = data.get_data_disk(os.path.join('..', 'data', 'synthetic', 'out'), ['x', 'y'])
    dvae.fit(generator, epochs=15, steps_per_epoch=spe)
