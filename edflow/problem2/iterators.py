import tensorflow as tf
from edflow.hooks.checkpoint_hooks.tf_checkpoint_hook import RestoreTFModelHook
from edflow.iterators.tf_iterator import TFHookedModelIterator
from edflow.iterators.tf_trainer import TFBaseTrainer
from edflow.project_manager import ProjectManager

from edflow.hooks.checkpoint_hooks.common import WaitForCheckpointHook

from edflow.hooks.checkpoint_hooks.common import get_latest_checkpoint
import time
import numpy as np


def loss(logits, labels):
    """Calculates the loss from the logits and the labels.
    Args:
    logits: Logits tensor, float - [batch_size, NUM_CLASSES].
    labels: Labels tensor, int32 - [batch_size].
    Returns:
    loss: Loss tensor of type float.
    """
    labels = tf.to_int64(labels)
    return tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)


class Trainer(TFBaseTrainer):
    def get_restore_variables(self):
        ''' nothing fancy here '''
        return super().get_restore_variables()


    def initialize(self, checkpoint_path = None):
        ''' in this case, we do not need to initialize anything special '''
        return super().initialize(checkpoint_path)



    def make_loss_ops(self):
        probs = self.model.outputs["probs"]
        logits = self.model.logits
        targets = self.model.inputs["target"]
        correct = tf.nn.in_top_k(probs, tf.cast(targets, tf.int32), k=1)
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        ce = loss(logits, targets)

        # losses are applied for each model
        # basically, we look for the string in the variables and update them with the loss provided here
        losses = dict()
        losses["model"] = ce

        # metrics for logging
        self.log_ops["acc"] = acc
        self.log_ops["ce"] = ce
        return losses


class ValidationIterator(TFHookedModelIterator):
    def __init__(self, *args, **kwargs):
        kwargs["desc"] = "Eval"
        kwargs["hook_freq"] = 1
        kwargs["num_epochs"] = 10
        super(InferenceIterator, self).__init__(*args, **kwargs)

        # wait for new checkpoint and restore

        self.setup()

    def step_ops(self):
        return tf.no_op()

    def make_feeds(self, batch):
        """Put global step into batches and add all extra required placeholders
        from batches."""
        feeds = super().make_feeds(batch)
        batch["global_step"] = self.get_global_step()
        for k, v in self.train_placeholders.items():
            if k in batch:
                feeds[v] = batch[k]
        return feeds

    def make_loss_ops(self):
        probs = self.model.outputs["probs"]
        logits = self.model.logits
        targets = self.model.inputs["target"]
        correct = tf.nn.in_top_k(probs, tf.cast(targets, tf.int32), k=1)
        acc = tf.reduce_mean(tf.cast(correct, tf.float32))

        ce = loss(logits, targets)

        # losses are applied for each model
        # basically, we look for the string in the variables and update them with the loss provided here
        losses = dict()
        losses["model"] = ce

        # metrics for logging
        self.log_ops["acc"] = acc
        self.log_ops["ce"] = ce
        return losses

    def run(self, fetches, feed_dict):
        get_global_step = fetches.pop("global_step")
        results = self.session.run(fetches, feed_dict=feed_dict)
        results["global_step"] = get_global_step()
        return results

    def setup(self):
        self.train_placeholders = dict()
        self.log_ops = dict()
        self.img_ops = dict()
        self.update_ops = list()
        _ = self.make_loss_ops()

        self.restore_variables = self.model.variables
        restorer = RestoreTFModelHook(
            variables=self.restore_variables,
            checkpoint_path=ProjectManager.checkpoints,
            global_step_setter=self.set_global_step,
        )

        waiter = ValidationHook(
            restorer=restorer,
            scalars=self.log_ops,
            logs=self.log_ops,
            images=self.img_ops,
            validation_root=ProjectManager.latest_eval, # TODO: how to get Project manager to create new eval and return it
            checkpoint_root=ProjectManager.checkpoints,
            eval_all=self.config.get("eval_all", False),
        )
        self.hooks.append(waiter)


class ValidationHook(WaitForCheckpointHook):
    def __init__(self,
                 restorer,
                 checkpoint_root,
                 filter_cond=lambda c: True,
                 interval=5,
                 add_sec=5,
                 eval_all=False,
                 scalars={},
                 histograms={},
                 images={},
                 logs={},
                 graph=None,
                 summary_tag="validation",
                 validation_root="logs",
                 *args, **kwargs):
        super(ValidationHook, self).__init__(checkpoint_root,
                                             filter_cond,
                                             interval,
                                             add_sec,
                                             eval_all,
                                             *args,
                                             **kwargs)
        self.restorer = restorer
        self.callback = self.restorer
        self.summary_tag = summary_tag

        self.logs = logs
        self.scalars = scalars

        self.fetch_dict = {"logs": logs, "images": images}

        self.log_interval = interval
        self.graph = graph
        self.validation_root = validation_root


    def look(self):
        """Loop until a new checkpoint is found."""
        self.logger.info("Waiting for new checkpoint.")
        while True:
            latest_checkpoint = get_latest_checkpoint(self.root, self.fcond)
            if (
                latest_checkpoint is not None
                and latest_checkpoint not in self.known_checkpoints
            ):
                self.known_checkpoints.add(latest_checkpoint)
                time.sleep(self.additional_wait)
                self.logger.info("Found new checkpoint: {}".format(latest_checkpoint))
                if self.callback is not None:
                    self.callback(latest_checkpoint)
                break
            time.sleep(self.sleep_interval)


    def before_epoch(self, epoch):
        # pass
        self.look()
        if epoch == 0:
            if self.graph is None:
                self.graph = tf.get_default_graph()

            self.writer = tf.summary.FileWriter(self.validation_root, self.graph)
        self.scalars_data = {s : list() for s in self.scalars}


    def before_step(self, batch_index, fetches, feeds, batch):
        if batch_index % self.log_interval == 0:
            fetches["logging"] = self.fetch_dict
        fetches["scalars"] = self.scalars


    def after_step(self, batch_index, last_results):
        # collect log data in list
        scalars = last_results["scalars"]
        for name in scalars.keys():
            self.scalars_data[name].append(scalars[name])
        if not "global_step" in self.scalars_data.keys():
            self.scalars_data["global_step"] = last_results["global_step"]

        if batch_index % self.log_interval == 0:
            step = last_results["global_step"]
            last_results = last_results["logging"]

            logs = last_results["logs"]
            for name in sorted(logs.keys()):
                self.logger.info("{}: {}".format(name, logs[name]))

            # TODO: what to do wich log images
            # for name, image_batch in last_results["images"].items():
            #     full_name = name + "_{:07}.png".format(step)
            #     save_path = os.path.join(self.root, full_name)
            #     plot_batch(image_batch, save_path)

            self.logger.info("project root: {}".format(self.root))


    def after_epoch(self, epoch):
        if self.writer:
            step = self.scalars_data["global_step"]
            for name, value_list in self.scalars_data.items():
                if not name == "global_step":
                    summary = tf.Summary(
                        value=[tf.Summary.Value(simple_value=np.mean(value_list), tag="{}/{}".format(self.summary_tag, name))])
                    self.writer.add_summary(summary, step)
            self.writer.flush()


