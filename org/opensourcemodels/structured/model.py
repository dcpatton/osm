import random
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras import Input, Model
from tensorflow.keras.layers import Dense

tfds.disable_progress_bar()

seed = 53
random.seed(seed)
tf.random.set_seed(seed)

print('TensorFlow version = ' + tf.__version__)

data = tfds.load('higgs', split='train', try_gcs=True)


def reformat_element(elem):
    features = [elem['jet_1_b-tag'], elem['jet_1_eta'], elem['jet_1_phi'], elem['jet_1_pt'], elem['jet_2_b-tag'],
                elem['jet_2_eta'], elem['jet_2_phi'], elem['jet_2_pt'], elem['jet_3_b-tag'], elem['jet_3_eta'],
                elem['jet_3_phi'], elem['jet_3_pt'], elem['jet_4_b-tag'], elem['jet_4_eta'], elem['jet_4_phi'],
                elem['jet_4_pt'], elem['lepton_eta'], elem['lepton_pT'], elem['lepton_phi'], elem['m_bb'],
                elem['m_jj'], elem['m_jjj'], elem['m_jlv'], elem['m_lv'], elem['m_wbb'], elem['m_wwbb'],
                elem['missing_energy_magnitude'], elem['missing_energy_phi']
                ]
    label = [elem['class_label']]
    return features, label


data = data.map(reformat_element, num_parallel_calls=tf.data.AUTOTUNE)
data = data.map(lambda x, y: (x, tf.cast(y, tf.int8)))

DATASET_SIZE = data.cardinality().numpy()
train_size = int(0.8 * DATASET_SIZE)
validation_size = int(0.10 * DATASET_SIZE)
test_size = int(0.10 * DATASET_SIZE)

data = data.shuffle(seed=seed, buffer_size=10240)
train = data.take(train_size)
test = data.skip(train_size)
validation = test.skip(test_size)
test = test.take(test_size)
print(train.cardinality().numpy())
print(validation.cardinality().numpy())
print(test.cardinality().numpy())

BATCH_SIZE = 1024
train = train.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
validation = validation.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)
test = test.batch(BATCH_SIZE).prefetch(tf.data.AUTOTUNE)

inp1 = Input(shape=(28, 1))
x = Dense(64, activation='swish')(inp1)
x = Dense(64, activation='swish')(x)
x = Dense(64, activation='swish')(x)
out1 = Dense(1, activation='sigmoid')(x)

model = Model(inputs=[inp1], outputs=[out1])
model.summary()

model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
              loss='bce',
              metrics=[tf.keras.metrics.BinaryAccuracy()])

es = tf.keras.callbacks.EarlyStopping(patience=9, verbose=1, restore_best_weights=True)

history = model.fit(train, verbose=2, epochs=100, validation_data=validation, callbacks=[es])

model.evaluate(test, verbose=1)

'''
 * baseline (28)  : loss: 0.6910 - binary_accuracy: 0.5302
 * baseline (21)  :
'''
