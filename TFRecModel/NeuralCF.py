import tensorflow as tf
import os

os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'

train_file_path = tf.keras.utils.get_file('training.csv', 'file:///Users/dongrenguang/my-git/rec-sys-store/src/main'
                                                          '/resources/webroot/ml-latest-small/trainingSamples.csv')
test_file_path = tf.keras.utils.get_file('test.csv', 'file:///Users/dongrenguang/my-git/rec-sys-store/src/main'
                                                     '/resources/webroot/ml-latest-small/testSamples.csv')

def get_dataset(file_path):
    """获取数据集"""
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=100,
        label_name='label',
        na_value='0',
        num_epochs=1,
        ignore_errors=True,
    )
    return dataset


train_dataset = get_dataset(train_file_path).shuffle(500)
test_dataset = get_dataset(test_file_path)

movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=193610)
movie_emb_col = tf.feature_column.embedding_column(movie_col, 20)

user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=611)
user_emb_col = tf.feature_column.embedding_column(user_col, 15)

inputs = {
    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
}

item_layer = tf.keras.layers.DenseFeatures([movie_emb_col])(inputs)
user_layer = tf.keras.layers.DenseFeatures([user_emb_col])(inputs)
interact_layer = tf.keras.layers.concatenate([item_layer, user_layer])
interact_layer = tf.keras.layers.Dense(64, activation='relu')(interact_layer)
interact_layer = tf.keras.layers.Dense(64, activation='relu')(interact_layer)
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(interact_layer)
model = tf.keras.Model(inputs, output_layer)

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=8)

test_loss, test_accuracy = model.evaluate(test_dataset)
print('\n\nTest Lost {}, Test Accuracy {}'.format(test_loss, test_accuracy))
