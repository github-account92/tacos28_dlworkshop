import argparse

import tensorflow as tf

from data import input_fn


N_CLASSES = 30


def model_fn_linear(features, labels, mode, params):
    logits = tf.layers.dense(features, N_CLASSES)
    base_lr = params["base_lr"]
    end_lr = params["end_lr"]
    decay_steps = params["decay_steps"]
    decay_power = params["decay_power"]

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {"logits": logits,
                       "probabilities": tf.nn.softmax(logits),
                       "input": features}
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

    cross_ent = tf.losses.sparse_softmax_cross_entropy(
        labels=labels, logits=logits)
    acc = tf.reduce_mean(
        tf.cast(tf.equal(tf.argmax(logits, axis=1, output_type=tf.int32), labels),
                tf.float32))
    tf.summary.scalar("accuracy", acc)

    if mode == tf.estimator.ModeKeys.TRAIN:
        gs = tf.train.get_global_step()
        lr = tf.train.polynomial_decay(base_lr, gs, decay_steps, end_lr,
                                       decay_power)
        tf.summary.scalar("learing_rate", lr)
        train_op = tf.train.AdamOptimizer(lr).minimize(cross_ent,
                                                       global_step=gs)
        return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                          train_op=train_op)

    eval_metric_ops = {"accuracy": tf.metrics.accuracy(labels, logits)}
    return tf.estimator.EstimatorSpec(mode=mode, loss=cross_ent,
                                      eval_metric_ops=eval_metric_ops)


parser = argparse.ArgumentParser()
parser.add_argument("mode")
parser.add_argument("base_path")
parser.add_argument("model_dir")
parser.add_argument("-B", "--batch_size",
                    type=int,
                    default=128)
parser.add_argument("-L", "--learning_rate",
                    nargs=2,
                    type=float,
                    default=[0.1, 0.000001],
                    help="Initial/final learning rate.")
parser.add_argument("-D", "--decay",
                    nargs=2,
                    type=float,
                    default=[100000, 2.0],
                    help="Decay steps and power.")
args = parser.parse_args()

prms = {"base_lr": args.learning_rate[0],
        "end_lr": args.learning_rate[1],
        "decay_steps": int(args.decay[0]),
        "decay_power": args.decay[1]}

tf.logging.set_verbosity(tf.logging.INFO)

est = tf.estimator.Estimator(model_fn=model_fn_linear,
                             model_dir=args.model_dir,
                             params=prms)

if args.mode == "train":
    est.train(input_fn=lambda: input_fn(args.base_path, "train", args.batch_size),
              steps=int(args.decay[0]))
elif args.mode == "predict":
    preds = est.predict(input_fn=lambda: input_fn(args.base_path, "dev", args.batch_size))
elif args.mode == "eval":
    print(est.evaluate(input_fn=lambda: input_fn(args.base_path, "dev", args.batch_size)))
