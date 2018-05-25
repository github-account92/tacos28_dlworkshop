import os
import pickle

import numpy as np
import tensorflow as tf


def input_fn_bow(base, dset, min_for_known, batchsize):
    # TODO choose test set maybe
    labels = []
    tweets = []
    if os.path.exists("vocab"):
        print("Found vocabulary, loading...")
        vocab = pickle.load(open("vocab", mode="rb"))

        with open(base, encoding="ISO-8859-1") as csv:
            for line in csv:
                label, text = line.split(",")[0][1], line.split(",")[-1][1:-2].split()
                if label == "0":
                    labels.append(0)
                elif label == "4":
                    labels.append(1)
                elif label == "2":
                    continue
                tweets.append(text)
    else:
        word_count = {}
        print("Gathering vocabulary...")
        with open(base, encoding="ISO-8859-1") as csv:
            for line in csv:
                label, text = line.split(",")[0][1], line.split(",")[-1][1:-2].split()
                if label == "0":
                    labels.append(0)
                elif label == "4":
                    labels.append(1)
                elif label == "2":
                    continue
                tweets.append(text)
                for word in text:
                    if word not in word_count:
                        word_count[word] = 0
                    word_count[word] += 1
        print("Got vocabulary of {} words".format(len(word_count)))

        print("Filtering words occurring less than {} "
              "times...".format(min_for_known))
        vocab = {}
        v_ind = 1
        for w, c in word_count.items():
            if c < min_for_known:
                vocab[w] = 0
            else:
                vocab[w] = v_ind
                v_ind += 1
        pickle.dump(vocab, open("vocab", mode="wb"))
        pickle.dump(word_count, open("word_count", mode="wb"))
        print("{} words remaining.".format(v_ind))
    num_known = max(vocab.values()) + 1

    def to_bow(ind, label):
        words = tweets[ind]
        bow = np.zeros(num_known, dtype=np.float32)
        for word in words:
            bow[vocab[word]] += 1
        return bow, [label]

    def gen():
        for ind, label in enumerate(labels):
            yield ind, label

    data = tf.data.Dataset.from_generator(gen,
                                          output_types=(tf.int32, tf.int32),
                                          output_shapes=((), ()))

    if dset == "train":
        data = data.apply(
            tf.contrib.data.shuffle_and_repeat(buffer_size=1600000))

    data = data.map(
        lambda ind, label: tuple(
            tf.py_func(to_bow, [ind, label], [tf.float32, tf.int32], False)))
    data = data.batch(batchsize)
    data = data.prefetch(2)

    iterator = data.make_one_shot_iterator()
    bow_batch, label_batch = iterator.get_next()
    bow_batch.set_shape([None, num_known])
    label_batch.set_shape([None, 1])
    return bow_batch, label_batch


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


if __name__ == "__main__":
    nexet = input_fn_bow(
        "data/testdata.manual.2009.06.14.csv", "dev", 5, 3)
    with tf.Session() as sess:
        while True:
            bobo, lab = sess.run(nexet)
            print(bobo.shape, lab)
            input()
