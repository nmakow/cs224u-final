import numpy as np
import tensorflow as tf
from tf_model_base import TfModelBase
from tensorflow.python.ops.rnn_cell import DropoutWrapper
import warnings
from modules import SelfAttn

__author__ = 'Chris Potts & Noah Makow'

# Ignore the TensorFlow warning
#   Converting sparse IndexedSlices to a dense Tensor of unknown shape.
#   This may consume a large amount of memory.
warnings.filterwarnings("ignore", category=UserWarning)


class TfRNNClassifier(TfModelBase):
    """Defines an RNN in which the final hidden state is used as
    the basis for a softmax classifier predicting a label:

    h_t = tanh(x_tW_xh + h_{t-1}W_hh)
    y   = softmax(h_nW_hy + b)

    t <= 1 <= n and the initial state h_0 is set to all 0s.

    Parameters
    ----------
    vocab : list
        The full vocabulary. `_convert_X` will convert the data provided
        to `fit` and `predict` methods into a list of indices into this
        list of items.
    embedding : 2d np.array or None
        If `None`, then a random embedding matrix is constructed.
        Otherwise, this should be a 2d array aligned row-wise with
        `vocab`, with each row giving the input representation for the
        corresponding word. For instance, to roughly duplicate what
        is done by default, one could do
            `np.array([np.random.randn(h) for _ in vocab])`
        where n is the embedding dimensionality (`embed_dim`).
    embed_dim : int
        Dimensionality of the inputs/embeddings. If `embedding`
        is supplied, then this value is set to be the same as its
        column dimensionality. Otherwise, this value is used to create
        the embedding Tensor (see `_define_embedding`).
    max_length : int
        Maximum sequence length.
    train_embedding : bool
        Whether to update the embedding matrix when training.
    cell_class : tf.nn.rnn_cell class
       The default is `tf.nn.rnn_cell.LSTMCell`. Other prominent options:
       `tf.nn.rnn_cell.BasicRNNCell`, and `tf.nn.rnn_cell.GRUCell`.
    hidden_activation : tf.nn activation
       E.g., tf.nn.relu, tf.nn.relu, tf.nn.selu.
    hidden_dim : int
        Dimensionality of the hidden layer.
    max_iter : int
        Maximum number of iterations allowed in training.
    eta : float
        Learning rate.
    tol : float
        Stopping criterion for the loss.
    """
    def __init__(self,
            vocab,
            embedding=None,
            embed_dim=50,
            max_length=20,
            train_embedding=True,
            cell_class=tf.nn.rnn_cell.LSTMCell,
            bidir_rnn=False,
            stacked=False,
            char_embed=False,
            word_length=15,
            num_char_filters=100,
            char_kernel_size=5,
            char_embed_dim=20,
	    self_attend=False,
            attention_dim=50,
            **kwargs):
        self.vocab = vocab
        self.vocab_size = len(vocab)
        self.embedding = embedding
        self.embed_dim = embed_dim
        self.max_length = max_length
        self.train_embedding = train_embedding
        self.cell_class = cell_class
        self.bidir_rnn = bidir_rnn
        self.stacked = stacked
        self.char_embed=char_embed
        self.word_length = word_length
        self.num_char_filters=num_char_filters
        self.char_kernel_size=char_kernel_size
        self.char_embed_dim=char_embed_dim
        self.self_attend = self_attend # use self-attention module after rnn-encoder
        self.attention_dim = attention_dim
        super(TfRNNClassifier, self).__init__(**kwargs)
        self.params += [
            'embedding', 'embed_dim', 'max_length', 'train_embedding']
        # self.char_vocab = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s", "t", "u", "v", "x", "y", "z", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9", ",", ".", "!", "(", ")", "[", "]", "%", "'", "-", "/", '"', "?", "$", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L", "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        
        self.char_vocab=[b'e', b't', b'a', b'o', b'i', b'n', b's', b'r', b'h', b'l', b'd', b'u', b'c', b'm', b'y', b'p', b'g', b'f', b'w', b'b', b'.', b'k', b'v', b',', b'I', b'"', b'T', b"'", b'A', b'S', b'W', b'C', b'E', b'P', b'O', b'N', b'H', b'!', b'1', b'-', b'0', b'R', b':', b'M', b')', b'x', b'B', b'D', b'(', b'2', b'U', b'F', b'L', b'?', b'j', b'G', b'Y', b'/', b'K', b'q', b'z', b'9', b'3', b'J', b'5', b'4', b'=', b'8', b'6', b'7', b'V', b';', b'_', b'|', b']', b'[', b'~', b'&', b'\\u2014', b'{', b'#', b'}', b'\\u2022', b'Q', b'%', b'*', b'Z', b'X', b'\\u2019', b'+', b'\\u2013', b'\\u201c', b'\\u201d', b'>', b'@', b'\\xb7', b'\\xe9', b'$', b'^', b'\\u200e', b'\\u2192', b'`', b'\\\\', b'\\xb4', b'<', b'\\u2018', b'\\u0430', b'\\u043e', b'\\xe1', b'\\u2026', b'\\u0435', b'\\u0438', b'\\xfc', b'\\xf6', b'\\u043d', b'\\u0627', b'\\xed', b'\\u03b1', b'\\u0440', b'\\u0131', b'\\xf3', b'\\u0644', b'\\u0442', b'\\u0441', b'\\xe4', b'\\u0107', b'\\u043a', b'\\xe7', b'\\u2665', b'\\u0432', b'\\u043b', b'\\u2190', b'\\u2666', b'\\u0101', b'\\u03b5', b'\\u25ba', b'\\u03c4', b'\\u2709', b'\\u260e', b'\\xa7', b'\\xb0', b'\\u0645', b'\\u0434', b'\\u266b', b'\\u0443', b'\\u0161', b'\\xa3', b'\\u03bf', b'\\xbb', b'\\u0648', b'\\u2663', b'\\u064a', b'\\u043f', b'\\u2605', b'\\u03c2', b'\\u2122', b'\\xab', b'\\u0646', b'\\u043c', b'\\u03bb', b'\\xa9', b'\\u0631', b'\\u266a', b'\\u03bd', b'\\u03b9', b'\\u015f', b'\\u2660', b'\\u2212', b'\\u263a', b'\\xe8', b'\\u0142', b'\\u2020', b'\\u010d', b'\\xe6', b'\\xe5', b'\\xe2', b'\\u0431', b'\\u03c1', b'\\u03b7', b'\\u0628', b'\\u2248', b'\\xe0', b'\\u200b', b'\\xf1', b'\\u0639', b'\\xa1', b'\\u014d', b'\\u044f', b'\\u03c9', b'\\u0647', b'\\xf8', b'\\xae', b'\\u0259', b'\\u0bcd', b'\\xfa', b'\\u2206', b'\\xeb', b'\\u03c0', b'\\u0437', b'\\u016b', b'\\xe3', b'\\u2207', b'\\u062a', b'\\xa2', b'\\u03c3', b'\\u062f', b'\\u0433', b'\\u0163', b'\\u044c', b'\\xba', b'\\u015b', b'\\u03ba', b'\\xdf', b'\\u0439', b'\\u30f3', b'\\u03b2', b'\\u03bc', b'\\u263c', b'\\u05d9', b'\\u093e', b'\\xaf', b'\\xd7', b'\\xea', b'\\u0930', b'\\u0633', b'\\u094d', b'\\u011f', b'\\u05d5', b'\\u03af', b'\\u2501', b'\\u03a9', b'\\u0391', b'\\u0395', b'\\u0629', b'\\u2500', b'\\xbf', b'\\u270d', b'\\u7adc', b'\\u262f', b'\\u0421', b'\\u042f', b'\\u0458', b'\\u05d4', b'\\u2550', b'\\xc9', b'\\u0250', b'\\u02c8', b'\\xa6', b'\\u05e8', b'\\u03b4', b'\\u0643', b'\\u012b', b'\\u3084', b'\\u0447', b'\\u03c5', b'\\u0394', b'\\u0144', b'\\u03b3', b'\\u20ac', b'\\u044b', b'\\u12ed', b'\\u03ac', b'\\u0160', b'\\u0641', b'\\u2764', b'\\xd8', b'\\xb2', b'\\u0642', b'\\xef', b'\\u0436', b'\\u0456', b'\\u062c', b'\\u20aa', b'\\u270e', b'\\u260f', b'\\u017c', b'\\u3044', b'\\u0623', b'\\u017e', b'\\u262e', b'\\xb1', b'\\u039c', b'\\xf4', b'\\u05ea', b'\\u062d', b'\\u026a', b'\\U00012073', b'\\u25d5', b'\\u21d4', b'\\u041c', b'\\u0412', b'\\u05de', b'\\xf0', b'\\uff61', b'\\xd6', b'\\u21d2', b'\\u6c34', b'\\u0448', b'\\u2260', b'\\u0119', b'\\u22c5', b'\\u05d0', b'\\u0634', b'\\u201e', b'\\u2704', b'\\u8a71', b'\\u570b', b'\\uff0c', b'\\u0bae', b'\\xc6', b'\\u05dc', b'\\xfd', b'\\u0445', b'\\u092e', b'\\xad', b'\\u30fc', b'\\u96f2', b'\\u2015', b'\\u0446', b'\\u093f', b'\\u306e', b'\\u2116', b'\\u2606', b'\\xa8', b'\\u0105', b'\\u039a', b'\\u0418', b'\\u0928', b'\\u03cc', b'\\u0103', b'\\xc7', b'\\u0938', b'\\u0b95', b'\\uff01', b'\\u06cc', b'\\u221e', b'\\xee', b'\\u05e9', b'\\u898b', b'\\u03c7', b'\\u2620', b'\\u2625', b'\\u9f99', b'\\u9023', b'\\u5b66', b'\\u0130', b'\\u03a3', b'\\u041f', b'\\u0924', b'\\u03c6', b'\\u2503', b'\\u2714', b'\\xa4', b'\\u7d61', b'\\u010c', b'\\u05d1']
        
        self.char_vocab_size = len(self.char_vocab) + 2 # for pad, unk
        self.char2id = {char: i+2 for i, char in enumerate(self.char_vocab)}
        self.PAD_ID = 0
        self.UNK_ID = 1

    def build_graph(self):
        self._define_embedding()

        self.inputs = tf.placeholder(
            tf.int32, [None, self.max_length])
        if self.char_embed:
            self.char_inputs = tf.placeholder(
                tf.int32, [None, self.max_length, self.word_length])

        self.ex_lengths = tf.placeholder(tf.int32, [None])

        # Outputs as usual:
        self.outputs = tf.placeholder(
            tf.float32, shape=[None, self.output_dim])

        # This converts the inputs to a list of lists of dense vector
        # representations:
        self.feats = tf.nn.embedding_lookup(
            self.embedding, self.inputs)
        if self.char_embed:
            # shape (?, max_len, word_len, char_embed_dim)
            self.char_feats = tf.nn.embedding_lookup(
                self.char_embedding, self.char_inputs)
            # shape (? * max_len, word_len, char_embed_dim)
            self.char_feats = tf.reshape(self.char_feats, 
                                         (-1, self.word_length, self.char_embed_dim))
            self.char_feats = tf.layers.conv1d(self.char_feats, 
                                               self.num_char_filters, 
                                               self.char_kernel_size, 
                                               padding="same")
            # shape (?, max_len, word_len, num_char_filters)
            self.char_feats = tf.reshape(self.char_feats, 
                                         (-1, self.max_length, self.word_length, self.num_char_filters))
            # shape (?, max_len, num_char_filters)
            self.char_feats = tf.reduce_max(self.char_feats, axis=2)
            # concatenate char embeds with word embeds to get final representation
            self.feats = tf.concat([self.feats, self.char_feats], 2)

        # Defines the RNN structure:
        self.cell = self.cell_class(
            self.hidden_dim, activation=self.hidden_activation)
        self.cell = DropoutWrapper(self.cell, self.dropout)
        # If bidirectional RNN used, define a second RNN cell
        # alternatively we could shared cells for fw/bw, but i think not for now.
        if self.bidir_rnn:
            self.bw_cell = self.cell_class(
                self.hidden_dim, activation=self.hidden_activation)
            self.bw_cell = DropoutWrapper(self.bw_cell, self.dropout)
            
        if self.stacked and self.bidir_rnn:
            self.cell2 = self.cell_class(
                self.hidden_dim, activation=self.hidden_activation)
            self.cell2 = DropoutWrapper(self.cell2, self.dropout)
            self.bw_cell2 = self.cell_class(
                self.hidden_dim, activation=self.hidden_activation)
            self.bw_cell2 = DropoutWrapper(self.bw_cell2, self.dropout)

        # Run the RNN:
        if self.bidir_rnn:
            with tf.variable_scope("lstm1", reuse=tf.AUTO_REUSE):
                outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                    self.cell,
                    self.bw_cell,
                    self.feats,
                    dtype=tf.float32,
                    sequence_length=self.ex_lengths)
                out = tf.concat(outputs, 1)
            
            if self.stacked:
                with tf.variable_scope("lstm2", reuse=tf.AUTO_REUSE):
                    outputs2, output_states2 = tf.nn.bidirectional_dynamic_rnn(
                        self.cell2,
                        self.bw_cell2,
                        out,
                        dtype=tf.float32,
                        sequence_length=self.ex_lengths)
            
            # let the last state be the concatenation of the fw and bw
            # final ``outputs''. Note that output_states[0] is the FW
            # (c, h) pair, and output_states[1] is the BW (c, h) pair where
            # c is the hidden state and h is the output
            if self.stacked:
                self.last = tf.concat([output_states2[0][1], output_states2[1][1]], 1)
            else:
                if self.cell_class == tf.nn.rnn_cell.LSTMCell:
                    self.last = tf.concat([output_states[0][1], output_states[1][1]], 1)
                elif self.cell_class == tf.nn.rnn_cell.GRUCell:
                    self.last = tf.concat(output_states, 1)
        else:
            outputs, state = tf.nn.dynamic_rnn(
                self.cell,
                self.feats,
                dtype=tf.float32,
                sequence_length=self.ex_lengths)

	# Attention Layer
        if self.self_attend:
            out = tf.concat(outputs, 1)
            print(out)
            self_attn_layer = SelfAttn(self.dropout, out.shape[-1], self.attention_dim)
            outputs = self_attn_layer.build_graph(out, self.ex_lengths)

        # How can I be sure that I have found the last true state? This
        # first option seems to work for all cell types but sometimes
        # leads to indexing errors and is in general pretty complex:
        #
        # self.last = self._get_last_non_masked(outputs, self.ex_lengths)
        #
        # This option is more reliable, but is it definitely getting
        # the final true state?
        #
        # Note that we set self.last above for the BiDir RNN case.
        if not self.bidir_rnn:
            self.last = self._get_final_state(self.cell, state)

        # Softmax classifier on the final hidden state:
        if self.bidir_rnn:
            self.W_hy = self.weight_init(
                2 * self.hidden_dim, self.output_dim, 'W_hy')
        else:
            self.W_hy = self.weight_init(
                self.hidden_dim, self.output_dim, 'W_hy')
        self.b_y = self.bias_init(self.output_dim, 'b_y')
        self.model = tf.matmul(self.last, self.W_hy) + self.b_y
       

    def train_dict(self, X, y):
        """Converts `X` to an np.array` using _convert_X` and feeds
        this to `inputs`, , and gets the true length of each example
        and passes it to `fit` as well. `y` is fed to `outputs`.

        Parameters
        ----------
        X : list of lists
        y : list

        Returns
        -------
        dict, list of int

        """
        inputs_X, ex_lengths = self._convert_X(X)
        if self.char_embed:
            char_X = self._convert_X_char(X)
            return {self.inputs: inputs_X, 
                self.char_inputs: char_X,
                self.ex_lengths: ex_lengths, 
                self.outputs: y}
        else:
            return {self.inputs: inputs_X, 
                self.ex_lengths: ex_lengths, 
                self.outputs: y}

    def test_dict(self, X):
        """Converts `X` to an np.array` using _convert_X` and feeds
        this to `inputs`, and gets the true length of each example and
        passes it to `fit` as well.

        Parameters
        ----------
        X : list of lists
        y : list

        Returns
        -------
        dict, list of int

        """
        inputs_X, ex_lengths = self._convert_X(X)
        if self.char_embed:
            char_X = self._convert_X_char(X)
            return {self.inputs: inputs_X, 
                    self.char_inputs: char_X,
                    self.ex_lengths: ex_lengths}
        else:
            return {self.inputs: inputs_X, 
                    self.ex_lengths: ex_lengths}

    def get_cost_function(self, **kwargs):
        """Uses `softmax_cross_entropy_with_logits` so the
        input should *not* have a softmax activation
        applied to it.
        """
        cost = tf.reduce_mean(
            tf.nn.sigmoid_cross_entropy_with_logits(
                logits=self.model, labels=self.outputs))
        tf.summary.scalar("loss", cost)
        return cost

    def prepare_output_data(self, y):
        """Format `y` so that Tensorflow can deal with it.

        Parameters
        ----------
        y : list of length num_classes, each is 0/1 if example is of that class

        Returns
        -------
        np.array with length the same as y and each row the
        length of the number of classes

        """
        self.classes = range(len(y[0])) # y[i] is len num_classes
        self.output_dim = len(self.classes)
        # just return y: already list of lists, shape (num_examples, num_classes)
        return y

    def predict_proba(self, X):
        """Return probabilistic predictions.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        np.array of predictions, dimension m x n, where m is the length
        of X and n is the number of classes

        """
        dataset = list(X)
        self.probs = tf.sigmoid(self.model)
        all_probs = np.zeros((len(X), self.output_dim))
        for i in range(0, len(dataset), self.batch_size):
            batch = dataset[i: i+self.batch_size]
            probs = self.sess.run(self.probs, feed_dict=self.test_dict(batch))
            all_probs[i: i+self.batch_size] = probs
        return all_probs
        # return self.sess.run(self.probs, feed_dict=self.test_dict(X))


    def predict(self, X):
        """Return classifier predictions.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        list

        """
        probs = self.predict_proba(X)
        return probs

    @staticmethod
    def _get_final_state(cell, state):
        """Get the final state from an RNN, managing differences in
        the TensorFlow API for cells.

        Parameters
        ----------
        cell : tf.nn.rnn_cell instance
        state : second argument returned by `tf.nn.dynamic_rnn`

        Returns
        -------
        Tensor

        """
        # If the cell is LSTMCell, then `state` is an `LSTMStateTuple`
        # and we want the second (output) Tensor -- see
        # https://www.tensorflow.org/api_docs/python/tf/contrib/rnn/LSTMStateTuple
        #
        if isinstance(cell, tf.nn.rnn_cell.LSTMCell):
            return state[1]
        else:
            # For other cell types, it seems we can just do this. I assume
            # that `state` is the last *true* state, not one of the
            # zero-padded ones (?).
            return state

    @staticmethod
    def _get_last_non_masked(outputs, lengths):
        """This method finds the last hidden state that is based on a
        non-null sequence element. It is adapted from

        https://danijar.com/variable-sequence-lengths-in-tensorflow/

        It's not currently being used, but it *might* be a more surefire
        way of ensuring that one retrieves the last true state. Compare
        with `_get_final_state.

        Parameters
        ----------
        outputs : a 3d Tensor of hidden states
        lengths : a 1d Tensor of ints

        Returns
        -------
        A 1d tensor, the last element of outputs that is based on a
        non-null input.

        """
        batch_size = tf.shape(outputs)[0]
        max_length = int(outputs.get_shape()[1])
        output_size = int(outputs.get_shape()[2])
        index = tf.range(0, batch_size) * max_length + (lengths - 1)
        flat = tf.reshape(outputs, [-1, output_size])
        last = tf.gather(flat, index)
        return last

    def _define_embedding(self):
        """Build the embedding matrix. If the user supplied a matrix, it
        is converted into a Tensor, else a random Tensor is built. This
        method sets `self.embedding` for use and returns None.
        """
        if type(self.embedding) == type(None):
            self.embedding = tf.Variable(
                tf.random_uniform(
                    [self.vocab_size, self.embed_dim], -1.0, 1.0),
                trainable=self.train_embedding)
        else:
            self.embedding = tf.Variable(
                initial_value=self.embedding,
                dtype=tf.float32,
                trainable=self.train_embedding)
            self.embed_dim = self.embedding.shape[1]
                
        if self.char_embed:
            self.char_embedding = tf.Variable(
                tf.random_uniform(
                    [self.char_vocab_size, self.char_embed_dim], -1.0, 1.0),
                trainable=self.train_embedding)

    def _convert_X(self, X):
        """Convert `X` to a list of list of indices into `self.vocab`,
        where all the lists have length `self.max_length`, which
        truncates the beginning of longer sequences and zero-pads the
        end of shorter sequences.

        Parameters
        ----------
        X : array-like
            The rows must be lists of objects in `self.vocab`.

        Returns
        -------
        np.array of int-type objects
            List of list of indices into `self.vocab`
        """
        new_X = np.zeros((len(X), self.max_length), dtype='int')
        ex_lengths = []
        index = dict(zip(self.vocab, range(len(self.vocab))))
        unk_index = index['$UNK']
        for i in range(new_X.shape[0]):
            ex_len = min([len(X[i]), self.max_length])
            ex_lengths.append(ex_len)
            vals = X[i][-self.max_length: ]
            vals = [index.get(w, unk_index) for w in vals]
            temp = np.zeros((self.max_length,), dtype='int')
            temp[0: len(vals)] = vals
            new_X[i] = temp
        return new_X, ex_lengths
    
    def _convert_X_char(self, X):
        """Convert X to a list of list of list of indices into
        self.char_vocab, where all the lists have length self.word_length,
        which truncates the beginning of longer words and zero-pads the end
        of shorter words.
        
        Parameters
        ----------
        X : array-like
            The rows must be lists of objects in `self.vocab`.

        Returns
        -------
        np.array of int-type objects: 
            shape (num_ex, self.max_length, self.word_length)
        """
        char_X = np.zeros((len(X), self.max_length, self.word_length), dtype='int')
        unk_index = self.UNK_ID
        for i in range(char_X.shape[0]):
            # Truncate to at most max_length tokens
            tokens = X[i][-self.max_length: ]
            for j,token in enumerate(tokens):
                # Truncate token to at most word_length tokens
                chars = token[-self.word_length: ]
                # Get ids for each char
                chars = [self.char2id.get(ch, self.UNK_ID) for ch in chars]
                # Copy into numpy array (0 padded)
                temp = np.zeros((self.word_length,), dtype='int')
                temp[0:len(chars)] = chars
                # Copy into the char_X matrix
                char_X[i, j, :] = temp
        return char_X
            


def simple_example():
    vocab = ['a', 'b', '$UNK']

    train = [
        [list('ab'), 'good'],
        [list('aab'), 'good'],
        [list('abb'), 'good'],
        [list('aabb'), 'good'],
        [list('ba'), 'bad'],
        [list('baa'), 'bad'],
        [list('bba'), 'bad'],
        [list('bbaa'), 'bad']]

    test = [
        [list('aaab'), 'good'],
        [list('baaa'), 'bad']]

    mod = TfRNNClassifier(
        vocab=vocab, max_iter=100, max_length=4)

    X, y = zip(*train)
    mod.fit(X, y)

    X_test, _ = zip(*test)
    print('\nPredictions:', mod.predict(X_test))


if __name__ == '__main__':
    simple_example()
