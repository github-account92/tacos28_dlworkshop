import pickle

import tensorflow as tf

from data import input_fn_bow, input_fn_raw, checkpoint_iterator


def model_fn_linear(features, labels, mode, params):
    base_lr = params["base_lr"]
    end_lr = params["end_lr"]
    decay_steps = params["decay_steps"]
    decay_power = params["decay_power"]
    reg_type = params["reg_type"]
    reg_coeff = params["reg_coeff"]
    mlp = params["mlp"]
    dropout = params["dropout"]
    rnn = params["rnn"]

    if reg_type == "l1":
        reg = lambda x: tf.norm(x, ord=1)
    elif reg_type == "l2":
        reg = lambda x: tf.norm(x, ord=2)
    else:
        reg = None

    if rnn:
        vocab = pickle.load(open("vocab", mode="rb"))
        one_hot = tf.one_hot(features["seq"], depth=max(vocab.values()) + 1)
        cell = tf.nn.rnn_cell.BasicRNNCell(512)
        features = tf.nn.dynamic_rnn(
            cell, one_hot, sequence_length=features["length"],
            dtype=tf.float32)[1]
    else:
        if mlp:
            features = tf.layers.dense(features, mlp, activation=tf.nn.relu,
                                       kernel_regularizer=reg)
        if dropout:
            features = tf.layers.dropout(
                features, training=mode == tf.estimator.ModeKeys.TRAIN)
    logit_layer = tf.layers.Dense(1, kernel_regularizer=reg)
    logits = logit_layer.apply(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"logits": logits,
                       "probabilities": tf.nn.sigmoid(logits),
                       "input": features}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_ent = tf.losses.sigmoid_cross_entropy(
        multi_class_labels=labels, logits=logits)
    loss = cross_ent
    reg_loss = tf.losses.get_regularization_loss()
    if reg_coeff:
        loss += reg_coeff * reg_loss

    labels_predict = tf.where(
        tf.greater_equal(logits, 0),
        tf.ones(tf.shape(logits), dtype=tf.int32),
        tf.zeros(tf.shape(logits), dtype=tf.int32))
    acc = tf.reduce_mean(
        tf.cast(tf.equal(labels_predict, labels),
                tf.float32))

    weights = logit_layer.trainable_weights[0]
    tf.summary.scalar("l1_loss", tf.norm(weights, ord=1))
    tf.summary.scalar("l2_loss", tf.norm(weights, ord=2))

    tf.summary.scalar("accuracy", acc)
    tf.summary.scalar("cross_ent", cross_ent)

    if mode == tf.estimator.ModeKeys.TRAIN:
        gs = tf.train.get_global_step()
        lr = tf.train.polynomial_decay(base_lr, gs, decay_steps, end_lr,
                                       decay_power)
        tf.summary.scalar("learning_rate", lr)
        train_op = tf.train.AdamOptimizer(lr).minimize(loss,
                                                       global_step=gs)
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                          train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels, labels_predict),
                       "cross_ent": tf.metrics.mean(cross_ent)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                      eval_metric_ops=eval_metric_ops)


def run(mode, base_path, model_dir,
        batch_size, learning_rate, decay, reg, min_for_known, mlp, dropout,
        rnn, use_char):
    prms = {"base_lr": learning_rate[0],
            "end_lr": learning_rate[1],
            "decay_steps": int(decay[0]),
            "decay_power": decay[1],
            "reg_type": reg[0],
            "reg_coeff": float(reg[1]),
            "mlp": mlp,
            "dropout": dropout,
            "rnn": rnn}

    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    keep_checkpoint_max=0,
                                    save_checkpoints_steps=5000)
    est = tf.estimator.Estimator(model_fn=model_fn_linear,
                                 config=config,
                                 params=prms)
    if rnn:
        inp_fn = lambda x, y, z: input_fn_raw(x, y, z, chars=use_char)
    else:
        inp_fn = input_fn_bow

    if mode == "train":
        inp = lambda: inp_fn(base_path, "train", min_for_known, batch_size)
    else:
        inp = lambda: inp_fn(base_path, "dev", min_for_known, batch_size)

    if mode == "train":
        est.train(input_fn=inp, steps=int(decay[0]))
    elif mode == "predict":
        return est.predict(input_fn=inp)
    elif mode == "eval":
        print(est.evaluate(input_fn=inp))
    elif mode == "eval-all":
        for ckpt in checkpoint_iterator(model_dir):
            print("Evaluating checkpoint {}...".format(ckpt))
            eval_results = est.evaluate(input_fn=inp)
            print("Evaluation results:\n", eval_results)
    elif mode == "return":
        return est
