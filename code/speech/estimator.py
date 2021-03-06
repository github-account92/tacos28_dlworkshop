import tensorflow as tf

from data import input_fn_raw, input_fn_tfr, checkpoint_iterator


N_CLASSES = 30


def model_fn_linear(features, labels, mode, params):
    base_lr = params["base_lr"]
    end_lr = params["end_lr"]
    decay_steps = params["decay_steps"]
    decay_power = params["decay_power"]
    reg_type = params["reg_type"]
    reg_coeff = params["reg_coeff"]
    mlp = params["mlp"]
    dropout = params["dropout"]
    conv = params["conv"]
    features_orig = features

    if reg_type == "l1":
        reg = lambda x: tf.norm(x, ord=1)
    elif reg_type == "l2":
        reg = lambda x: tf.norm(x, ord=2)
    else:
        reg = None

    if conv:
        features = tf.expand_dims(features, -1)
        features = tf.layers.conv2d(features, 64, 5, padding="same",
                                    activation=None,
                                    kernel_regularizer=reg)
        features = tf.layers.batch_normalization(
            features, training=mode == tf.estimator.ModeKeys.TRAIN)
        features = tf.nn.relu(features)
        features = tf.layers.max_pooling2d(features, 2, 2, padding="same")
        pool1 = features
        features = tf.layers.conv2d(features, 128, 5, padding="same",
                                    activation=None,
                                    kernel_regularizer=reg)
        features = tf.layers.batch_normalization(
            features, training=mode == tf.estimator.ModeKeys.TRAIN)
        features = tf.nn.relu(features)
        features = tf.layers.max_pooling2d(features, 2, 2, padding="same")
        pool2 = features

    features = tf.layers.flatten(features)
    if mlp:
        features = tf.layers.dense(features, mlp, activation=None,
                                   kernel_regularizer=reg)
        features = tf.layers.batch_normalization(
            features, training=mode == tf.estimator.ModeKeys.TRAIN)
        features = tf.nn.relu(features)
    if dropout:
        features = tf.layers.dropout(
            features, training=mode == tf.estimator.ModeKeys.TRAIN)
    logit_layer = tf.layers.Dense(N_CLASSES, kernel_regularizer=reg)
    logits = logit_layer.apply(features)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"logits": logits,
                       "probabilities": tf.nn.softmax(logits),
                       "input": features_orig}
        if conv:
            predictions.update({"pool1": pool1, "pool2": pool2})
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_ent = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)
    loss = cross_ent
    reg_loss = tf.losses.get_regularization_loss()
    if reg_coeff:
        loss += reg_coeff * reg_loss

    labels_predict = tf.argmax(logits, axis=1, output_type=tf.int32)
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

        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = tf.train.AdamOptimizer(lr).minimize(loss,
                                                           global_step=gs)
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                          train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels, labels_predict),
                       "cross_ent": tf.metrics.mean(cross_ent)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                      eval_metric_ops=eval_metric_ops)


def run(mode, base_path, model_dir,
        batch_size, learning_rate, decay, reg, mel, mlp, dropout, conv):
    prms = {"base_lr": learning_rate[0],
            "end_lr": learning_rate[1],
            "decay_steps": int(decay[0]),
            "decay_power": decay[1],
            "reg_type": reg[0],
            "reg_coeff": float(reg[1]),
            "mlp": mlp,
            "dropout": dropout,
            "conv": conv}

    tf.logging.set_verbosity(tf.logging.INFO)

    config = tf.estimator.RunConfig(model_dir=model_dir,
                                    keep_checkpoint_max=0,
                                    save_checkpoints_steps=5000)
    est = tf.estimator.Estimator(model_fn=model_fn_linear,
                                 config=config,
                                 params=prms)

    if mel:
        input_fn = input_fn_tfr
    else:
        input_fn = input_fn_raw

    if mode == "train":
        inp = lambda: input_fn(base_path, "train", batch_size)
    else:
        inp = lambda: input_fn(base_path, "dev", batch_size)

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
