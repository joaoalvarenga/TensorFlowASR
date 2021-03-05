import tensorflow as tf

from tensorflow_asr.configs.config import Config
from tensorflow_asr.featurizers.text_featurizers import SubwordFeaturizer
from tensorflow_asr.models.layers.embedding import Embedding
from tensorflow_asr.utils.utils import get_rnn


class ExternalLM(tf.keras.Model):
    def __init__(self,
                 vocabulary_size: int,
                 embed_dim: int = 512,
                 embed_dropout: float = 0,
                 num_rnns: int = 3,
                 rnn_units: int = 4096,
                 rnn_type: str = 'lstm',
                 layer_norm: bool = True,
                 kernel_regularizer=None,
                 bias_regularizer=None,
                 name='ExternalLM'):
        super(ExternalLM, self).__init__(name=name)
        self.embed = Embedding(vocabulary_size, embed_dim,
                               regularizer=kernel_regularizer, name=f"{name}_embedding")
        self.do = tf.keras.layers.Dropout(embed_dropout, name=f"{name}_dropout")
        RNN = get_rnn(rnn_type)
        self.rnns = []
        for i in range(num_rnns):
            rnn = RNN(
                units=rnn_units, return_sequences=True,
                name=f"{name}_lstm_{i}", return_state=True,
                kernel_regularizer=kernel_regularizer,
                bias_regularizer=bias_regularizer
            )
            if layer_norm:
                ln = tf.keras.layers.LayerNormalization(name=f"{name}_ln_{i}")
            else:
                ln = None
            self.rnns.append({"rnn": rnn, "ln": ln, "projection": None})
        self.ffn_out = tf.keras.layers.Dense(
            vocabulary_size, name=f"{name}_vocab",
            kernel_regularizer=kernel_regularizer,
            bias_regularizer=bias_regularizer
        )

    def get_initial_state(self):
        """Get zeros states

        Returns:
            tf.Tensor: states having shape [num_rnns, 1 or 2, B, P]
        """
        states = []
        for rnn in self.rnns:
            states.append(
                tf.stack(
                    rnn["rnn"].get_initial_state(
                        tf.zeros([1, 1, 1], dtype=tf.float32)
                    ), axis=0
                )
            )
        return tf.stack(states, axis=0)

    def call(self, inputs, states=None, training=False):
        # inputs has shape [B, U]
        # use tf.gather_nd instead of tf.gather for tflite conversion
        outputs = self.embed(inputs, training=training)
        outputs = self.do(outputs, training=training)
        if states is None:
            states = self.get_initial_state()
        for i, rnn in enumerate(self.rnns):
            outputs = rnn["rnn"](outputs, training=training)
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=training)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=training)
        outputs = self.ffn_out(outputs, training=training)
        return outputs

    def recognize(self, inputs, states):
        """Recognize function for prediction network

        Args:
            inputs (tf.Tensor): shape [1, 1]
            states (tf.Tensor): shape [num_lstms, 2, B, P]

        Returns:
            tf.Tensor: outputs with shape [1, 1, P]
            tf.Tensor: new states with shape [num_lstms, 2, 1, P]
        """
        outputs = self.embed(inputs, training=False)
        outputs = self.do(outputs, training=False)
        new_states = []
        for i, rnn in enumerate(self.rnns):
            outputs = rnn["rnn"](outputs, training=False,
                                 initial_state=tf.unstack(states[i], axis=0))
            new_states.append(tf.stack(outputs[1:]))
            outputs = outputs[0]
            if rnn["ln"] is not None:
                outputs = rnn["ln"](outputs, training=False)
            if rnn["projection"] is not None:
                outputs = rnn["projection"](outputs, training=False)
        outputs = self.ffn_out(outputs, training=False)
        return outputs, tf.stack(new_states, axis=0)

    def get_config(self):
        conf = self.embed.get_config()
        conf.update(self.do.get_config())
        for rnn in self.rnns:
            conf.update(rnn["rnn"].get_config())
            if rnn["ln"] is not None:
                conf.update(rnn["ln"].get_config())
            if rnn["projection"] is not None:
                conf.update(rnn["projection"].get_config())
        return conf


def perplexity(y_true, y_pred):
    return tf.keras.backend.exp(tf.keras.backend.mean(tf.keras.backend.sparse_categorical_crossentropy(y_true, y_pred)))



def process(text):
    encoded_output = subword.extract(text.decode('utf-8'))
    encoded_input = subword.prepand_blank(encoded_output)
    encoded_output = tf.concat([encoded_output, [subword.blank]], axis=0)
    assert encoded_input.shape == encoded_output.shape
    return encoded_input, encoded_output

@tf.function
def parse(record):
    return tf.numpy_function(
        process,
        inp=[record],
        Tout=[tf.int32, tf.int32]
    )

config = Config('config.yml', learning=True)
subword = SubwordFeaturizer.load_from_file(config.decoder_config,
                                           '/home/joao/mestrado/datasets/conformer_subwords.subwords')
print(subword.num_classes)
batch_size = 32
dataset = tf.data.TextLineDataset('sample_data.txt')
dataset = dataset.map(parse)
dataset = dataset.cache()
# dataset = dataset.batch(batch_size, drop_remainder=True)
dataset = dataset.padded_batch(
            batch_size=batch_size,
            padded_shapes=(
                tf.TensorShape([None]),
                tf.TensorShape([None])
            ),
            padding_values=(subword.blank, subword.blank),
            drop_remainder=True
        )
model = ExternalLM(subword.num_classes, rnn_units=320)
model.compile(optimizer='adam', loss="sparse_categorical_crossentropy", metrics=[perplexity])
history = model.fit(dataset, epochs=100)
x = tf.expand_dims(tf.concat(([0], subword.extract('a subida')), 0), axis=0)
print(x)
y = model.predict(x)
y = tf.argmax(y, axis=-1, output_type=tf.int32)
print(y)
print(subword.iextract(y))
