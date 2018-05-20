import os

import librosa
import numpy as np
import tensorflow as tf


def input_fn(base, dset, batchsize):
    data = tf.data.Dataset.from_generator(make_iterator(base, dset),
                                          output_types=(tf.float32, tf.int32))
    if dset == "train":
        data = data.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=1000))
    data = data.batch(batchsize)
    data = data.prefetch(2)

    iterator = data.make_one_shot_iterator()
    return iterator.get_next()


def make_iterator(base, dset):
    """Prepare iterator over dataset.

    Parameters:
        base: Base path with the dataset.
        dset: string, one of train, dev or test. Which set to extract.
    """
    val = set(open(os.path.join(base, "validation_list.txt")).readlines())
    test = set(open(os.path.join(base, "testing_list.txt")).readlines())
    if dset == "train":
        excludes = val.union(test)
        permitted = []
    elif dset == "dev":
        permitted = val
        excludes = []
    elif dset == "test":
        permitted = test
        excludes = []
    else:
        raise ValueError("Invalid subset lol: {}".format(dset))
    excludes = set(e.strip() for e in excludes)
    permitted = set(p.strip() for p in permitted)

    folders = [f for f in os.listdir(base) if
               os.path.isdir(os.path.join(base, f)) and f != "_background_noise_"]
    class_map = dict((cl, ind) for (ind, cl) in enumerate(folders))

    def gen():
        for label in folders:
            samples = os.listdir(os.path.join(base, label))
            for sample in samples:
                if (excludes and not "/".join([label, sample]) in excludes) or \
                        (permitted and "/".join([label, sample]) in permitted):
                    audio, sr = librosa.load(
                        os.path.join(base, label, sample), sr=None)
                    if len(audio) != 16000:
                        audio = np.pad(audio, (0, 16000-len(audio)),
                                       mode="constant")
                    yield audio, class_map[label]

    return gen
