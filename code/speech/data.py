import os

import librosa
import numpy as np
import tensorflow as tf


def input_fn_raw(base, dset, batchsize):
    data = tf.data.Dataset.from_generator(make_iterator(base, dset),
                                          output_types=(tf.float32, tf.int32),
                                          output_shapes=((16000,), ()))
    if dset == "train":
        data = data.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=60000))
    data = data.batch(batchsize)
    data = data.prefetch(2)

    iterator = data.make_one_shot_iterator()
    return iterator.get_next()


def input_fn_tfr(base, dset, batchsize):
    data = tf.data.TFRecordDataset(base)
    data = data.map(parse_tfr)
    if dset == "train":
        data = data.apply(
                tf.contrib.data.shuffle_and_repeat(buffer_size=60000))
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


def make_tfrecord(out_path, base, dset):
    iterator = make_iterator(base, dset)()
    with tf.python_io.TFRecordWriter(out_path + dset + ".tfrecords") as writer:
        for seq, label in iterator:
            mel = librosa.feature.melspectrogram(seq, sr=16000, n_fft=400,
                                                 hop_length=160)
            mel = np.log(mel + 1e-10)
            print(mel.min(), mel.max(), mel.shape)
            input()
            tfex = tf.train.Example(
                features=tf.train.Features(
                    feature={"audio": tf.train.Feature(
                        float_list=tf.train.FloatList(value=mel.flatten())),
                             "label": tf.train.Feature(
                        int64_list=tf.train.Int64List(value=[label]))}))
            writer.write(tfex.SerializeToString())


def checkpoint_iterator(ckpt_folder):
    """Iterates over checkpoints in order and returns them.

    This modifies the "checkpoint meta file" directly which might not be the
    smartest way to do it.
    Note that this file yields checkpoint names for convenience, but the main
    function is actually the modification of the meta file.

    Parameters:
        ckpt_folder: Path to folder that has all the checkpoints. Usually the
                     estimator's model_dir. Also needs to contain a file called
                     "checkpoint" that acts as the "meta file".

    Yields:
        Paths to checkpoints, in order.
    """
    # we store the original text to re-write it
    try:
        with open(os.path.join(ckpt_folder, "checkpoint")) as ckpt_file:
            next(ckpt_file)
            orig = ckpt_file.read()
    except:  # the file might be empty because reasons...
        orig = ""

    # get all the checkpoints
    # we can't rely on the meta file (doesn't store permanent checkpoints :()
    # so we check the folder instead.
    ckpts = set()
    for file in os.listdir(ckpt_folder):
        if file.split("-")[0] == "model.ckpt":
            ckpts.add(int(file.split("-")[1].split(".")[0]))
    ckpts = sorted(list(ckpts))
    ckpts = ["\"model.ckpt-" + str(ckpt) + "\"" for ckpt in ckpts]

    # fill them in one-by-one and leave
    for ckpt in ckpts:
        with open(os.path.join(ckpt_folder, "checkpoint"),
                  mode="w") as ckpt_file:
            ckpt_file.write("model_checkpoint_path: " + ckpt + "\n")
            ckpt_file.write(orig)
        yield ckpt


def parse_tfr(example_proto):
    features = {"audio": tf.FixedLenFeature((128*101,), tf.float32),
                "label": tf.FixedLenFeature((), tf.int64)}
    parsed_features = tf.parse_single_example(example_proto, features)
    return (tf.reshape(parsed_features["audio"], [128, 101]),
            tf.cast(parsed_features["label"], tf.int32))


if __name__ == "__main__":
    make_tfrecord("test", "data_speech_commands_v0.01", "train")
