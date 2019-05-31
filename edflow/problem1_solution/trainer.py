from edflow.iterators.trainer import TFBaseTrainer

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