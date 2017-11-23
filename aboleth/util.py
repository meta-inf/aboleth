"""Package helper utilities."""
import tensorflow as tf
import numpy as np

from tensorflow.python.framework import meta_graph
from tensorflow.python.framework import ops
from tensorflow.python.training import training_util
from tensorflow.python.training.session_run_hook import SessionRunArgs

from aboleth.random import endless_permutations


class CurrentBestCheckpointSaverHook(tf.train.CheckpointSaverHook):

    def __init__(self, tensor, *args, **kwargs):
        self.tensor = tensor
        self.current_min_tensor_value = None
        super(CurrentBestCheckpointSaverHook, self).__init__(*args, **kwargs)

    def before_run(self, run_context):
        if self._timer.last_triggered_step() is None:
            training_util.write_graph(
                ops.get_default_graph().as_graph_def(add_shapes=True),
                self._checkpoint_dir,
                "graph.pbtxt")
            saver_def = self._get_saver().saver_def if self._get_saver() \
                else None
            graph = ops.get_default_graph()
            meta_graph_def = meta_graph.create_meta_graph_def(
                graph_def=graph.as_graph_def(add_shapes=True),
                saver_def=saver_def)
            self._summary_writer.add_graph(graph)
            self._summary_writer.add_meta_graph(meta_graph_def)

        return SessionRunArgs(self._global_step_tensor, self.tensor)

    def after_run(self, run_context, run_values):
        stale_global_step, tensor_value = run_values.results
        if self._timer.should_trigger_for_step(stale_global_step+1):
            # get the real value after train op.
            global_step = run_context.session.run(self._global_step_tensor)
            if self._timer.should_trigger_for_step(global_step):
                self._timer.update_last_triggered_step(global_step)

                if (self.current_min_tensor_value is None) or \
                   (tensor_value < self.current_min_tensor_value):

                    self.current_min_tensor_value = tensor_value
                    self._save(run_context.session, global_step)


def pos(X, minval=1e-15):
    r"""Constrain a ``tf.Variable`` to be positive only.

    At the moment this is implemented as:

        :math:`\max(|\mathbf{X}|, \text{minval})`

    This is fast and does not result in vanishing gradients, but will lead to
    non-smooth gradients and more local minima. In practice we haven't noticed
    this being a problem.

    Parameters
    ----------
    X : Tensor
        any Tensor in which all elements will be made positive.
    minval : float
        the minimum "positive" value the resulting tensor will have.

    Returns
    -------
    X : Tensor
        a tensor the same shape as the input ``X`` but positively constrained.

    Examples
    --------
    >>> X = tf.constant(np.array([1.0, -1.0, 0.0]))
    >>> Xp = pos(X)
    >>> with tf.Session():
    ...     xp = Xp.eval()
    >>> xp
    array([  1.00000000e+00,   1.00000000e+00,   1.00000000e-15])

    """
    # Other alternatives could be:
    # Xp = tf.exp(X)  # Medium speed, but gradients tend to explode
    # Xp = tf.nn.softplus(X)  # Slow but well behaved!
    Xp = tf.maximum(tf.abs(X), minval)  # Faster, but more local optima
    return Xp


def batch(feed_dict, batch_size, n_iter=10000, N_=None):
    r"""Create random batches for Stochastic gradients.

    Feed dict data generator for SGD that will yeild random batches for a
    a defined number of iterations, which can be infinite. This generator makes
    consecutive passes through the data, drawing without replacement on each
    pass.

    Parameters
    ----------
    feed_dict : dict of ndarrays
        The data with ``{tf.placeholder: data}`` entries. This assumes all
        items have the *same* length!
    batch_size : int
        number of data points in each batch.
    n_iter : int, optional
        The number of iterations
    N_ : tf.placeholder (int), optional
        Place holder for the size of the dataset. This will be fed to an
        algorithm.

    Yields
    ------
    dict:
        with each element an array length ``batch_size``, i.e. a subset of
        data, and an element for ``N_``. Use this as your feed-dict when
        evaluating a loss, training, etc.

    """
    N = __data_len(feed_dict)
    perms = endless_permutations(N)

    i = 0
    while i < n_iter:
        i += 1
        ind = np.array([next(perms) for _ in range(batch_size)])
        batch_dict = {k: v[ind] for k, v in feed_dict.items()}
        if N_ is not None:
            batch_dict[N_] = N
        yield batch_dict


def batch_prediction(feed_dict, batch_size):
    r"""Split the data in a feed_dict into contiguous batches for prediction.

    Parameters
    ----------
    feed_dict : dict of ndarrays
        The data with ``{tf.placeholder: data}`` entries. This assumes all
        items have the *same* length!
    batch_size : int
        number of data points in each batch.

    Yields
    ------
    ndarray :
        an array of shape approximately (``batch_size``,) of indices into the
        original data for the current batch
    dict :
        with each element an array length ``batch_size``, i.e. a subset of
        data. Use this as your feed-dict when evaluating a model, prediction,
        etc.

    Note
    ----
    The exact size of the batch may not be ``batch_size``, but the nearest size
    that splits the size of the data most evenly.

    """
    N = __data_len(feed_dict)
    n_batches = max(np.round(N / batch_size), 1)
    batch_inds = np.array_split(np.arange(N, dtype=int), n_batches)

    for ind in batch_inds:
        batch_dict = {k: v[ind] for k, v in feed_dict.items()}
        yield ind, batch_dict


def __data_len(feed_dict):
    N = feed_dict[list(feed_dict.keys())[0]].shape[0]
    return N
