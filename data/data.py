"""Dataloader generator.

We insert a list of directories, where each directory contains a certain variable of the dataset split into chunks,
saved as numpy arrays (e.g., 0.npy, 1.npy, .., n.npy).

Saving an entire dataset to such chunks works as follows:
1. Shuffle all data.
2. Split data into n separate equal-size pieces.
3. Save all these pieces to disk.
Note that if we have multiple data 'items', i.e., an x and a y, we expect indices to match between those (they should
be shuffled together).

Gathering data for one epoch then works as follows:
1. Shuffle the order of all chunks.
2. Load a chunk into memory, shuffle its internal order.
3. Process entire chunk in batches (specified by batch_size).
4. When done processing chunk, load next chunk from the 'shuffled order' and repeat (step 2.)
5. When done processing all chunks, epoch is done.

Perhaps there's more efficient approaches to do this (e.g., multi-threaded, using built in PyTorch / TensorFlow
functions), but this works and it's not the most important, so it's not part of the rewrite ;-)
"""

# TODO: Make sure Change Disc y-labels are [1, 0] = False, [0, 1] = True (so one-hot of 0/1, swapped in existing data)
# TODO: This is now the case for presaved data --> for code creating this data, ensure this!
# TODO: Also, for converting pre-trained models from old codebase, ensure the prediction matches (fix dense layer)
import numpy as np
import math
import glob
import os
import tensorflow as tf


# load chunks from disk
def get_batch_list_chunk(batch_size, data_list, start_idx, idx_order, tuple_split=-1):
    # we process batch_size elements from the chunk starting at start_idx, using idx_order
    # to determine which elements we pick
    batch_list = []
    num_data = len(data_list)
    for i in range(num_data):
        dataset = data_list[i]
        N = dataset.shape[0]
        datashape = dataset.shape[1:]
        if not datashape:
            do_reshape = False
        else:
            do_reshape = True
        batch = np.zeros((batch_size, *datashape), dtype=np.float32)
        for j in range(batch_size):
            if j + start_idx >= N:  # loop back around for last few elements
                idx = j + start_idx - N
            else:
                idx = j + start_idx
            if do_reshape:
                batch[j, ...] = dataset[idx_order[idx]].reshape(*datashape)
            else:
                batch[j, ...] = dataset[idx_order[idx]]
        batch_list.append(batch)
    if tuple_split == -1:
        return tuple(batch_list)
    else:
        split_a = batch_list[:tuple_split]
        split_b = batch_list[tuple_split:]
        split_a = tuple(split_a)
        split_b = tuple(split_b)
        return split_a, split_b


# x_dir, x_mask_dir, x_hl_dir, y_dir, p_x_dir, p_x_mask_dir, p_x_hl_dir, p_y_dir
def batch_generator_list_disk(batch_size, dirs, num_chunk, tuple_split=-1):
    # tf.data.Dataset.from_generator converts str arguments to bytes for some reason; decode
    if not isinstance(dirs[0], str):
        dirs = [x.decode('utf-8') for x in dirs]

    # are we at a new epoch -> randomize chunk order, load all chunks again
    new_epoch = True
    # are we at a new chunk -> randomize index-within-chunk order, process chunk
    new_chunk = True

    # current chunk we have loaded in memory
    cached_chunks = []
    num_data = len(dirs)

    # size of every chunk (we do not know this until we load one)
    chunk_size = None
    while True:
        if new_epoch:
            # reset counters as we are processing a new epoch
            chunk_idx = 0
            chunk_order = np.random.permutation(num_chunk)
            new_chunk = True
            new_epoch = False
        if new_chunk:
            # load data for the assigned chunk from disk
            cached_chunks = []
            for i in range(num_data):
                data = np.load(os.path.join(dirs[i], '{}.npy').format(chunk_order[chunk_idx]))
                cached_chunks.append(data)

            # shuffle the indices within the chunk -> randomize our within-chunk batches
            chunk_size = cached_chunks[0].shape[0]
            idx_order = np.random.permutation(chunk_size)
            element_idx = 0
            new_chunk = False
        # process batch
        yield get_batch_list_chunk(batch_size, cached_chunks, element_idx, idx_order, tuple_split)
        element_idx += batch_size

        if element_idx >= chunk_size:  # go to next chunk; we processed all indices
            chunk_idx += 1
            new_chunk = True
            if chunk_idx >= num_chunk:  # go to next epoch; we processed all chunks
                print('Epoch completed')
                new_epoch = True


def get_var_info(dir_base, dir_name):
    chunks = glob.glob(os.path.join(dir_base, dir_name, '*.npy'))
    chunk_idx = [int(os.path.basename(f)[:-len('.npy')]) for f in chunks]
    chunk_idx.sort()
    num_chunk = max(chunk_idx) + 1
    assert chunk_idx == list(range(num_chunk))  # make sure we have all chunks
    with open(os.path.join(dir_base, dir_name, 'size.txt'), 'r') as f:
        total_elem = f.read()
        total_elem = int(total_elem)
    return num_chunk, total_elem


def get_data_disk(dir_base, dir_names, batch_size=128):
    num_chunk, total_elem = get_var_info(dir_base, dir_names[0])
    for d in dir_names:
        assert num_chunk, total_elem == get_var_info(dir_base, d)

    chunk_size = math.floor(total_elem / num_chunk)
    steps_per_epoch = math.ceil(chunk_size / batch_size) * num_chunk
    full_paths = [os.path.join(dir_base, x) for x in dir_names]
    args = [batch_size, full_paths, num_chunk]
    output_types = tuple([tf.float32 for x in range(len(full_paths))])
    data_generator = tf.data.Dataset.from_generator(batch_generator_list_disk, args=args, output_types=output_types)

    return data_generator, steps_per_epoch


if __name__ == '__main__':
    # simple test
    generator, spe = get_data_disk(os.path.join('synthetic', 'out'), ['x', 'y'])
    for i, batch in enumerate(generator):
        if i == spe:  # done with epoch
            break
        print('batch of {}'.format(tuple([x.shape for x in batch])))
