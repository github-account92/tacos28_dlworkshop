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

        with open(base) as csv:
            for line in csv:
                label, text = line.split(",")[0][1], line.split(",")[-1][1:-2].split()
                if label == "0":
                    labels.append(0)
                elif label == "4":
                    labels.append(1)
                tweets.append(text)
    else:
        word_count = {}
        print("Gathering vocabulary...")
        with open(base) as csv:
            for line in csv:
                label, text = line.split(",")[0][1], line.split(",")[-1][1:-2].split()
                if label == "0":
                    labels.append(0)
                elif label == "4":
                    labels.append(1)
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
        bow = np.zeros(num_known, dtype=np.int32)
        for word in words:
            bow[vocab[word]] += 1
        return bow, label

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
            tf.py_func(to_bow, [ind, label], [tf.int32, tf.int32], False)))
    data = data.batch(batchsize)
    data = data.prefetch(2)

    iterator = data.make_one_shot_iterator()
    return iterator.get_next()


if __name__ == "__main__":
    nexet = input_fn_bow(
        "data/training.1600000.processed.noemoticon.csv", "dev", 5, 3)
    with tf.Session() as sess:
        while True:
            bobo, label = sess.run(nexet)
            print(bobo.shape, label)
            input()
