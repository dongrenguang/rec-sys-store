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


# 处理类别型特征
genre_vocab = ['Action', 'Comedy', 'Drama', 'Adventure', 'Crime', 'Animation', 'Horror',
               'Children', 'Documentary', 'Mystery', 'Thriller', 'Sci-Fi', 'Fantasy', 'Western',
               'Musical', 'Romance', 'Film-Noir', '(no genres listed)', 'War', 'IMAX']
CATEGORIES = {
    'movieGenre1': genre_vocab,
    'movieGenre2': genre_vocab,
    'movieGenre3': genre_vocab,
    'userGenre1': genre_vocab,
    'userGenre2': genre_vocab,
    'userGenre3': genre_vocab,
    'userGenre4': genre_vocab,
    'userGenre5': genre_vocab
}
categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(key=feature, vocabulary_list=vocab)
    emb_col = tf.feature_column.embedding_column(cat_col, 10)
    categorical_columns.append(emb_col)

movie_col = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=193610)
movie_emb_col = tf.feature_column.embedding_column(movie_col, 20)
categorical_columns.append(movie_emb_col)

user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=611)
user_emb_col = tf.feature_column.embedding_column(user_col, 15)
categorical_columns.append(user_emb_col)

# 处理数值型特征
numeric_columns = [
    tf.feature_column.numeric_column('releaseYear'),
    tf.feature_column.numeric_column('movieRatingCount'),
    tf.feature_column.numeric_column('movieAvgRating'),
    tf.feature_column.numeric_column('movieRatingStddev'),
    tf.feature_column.numeric_column('userRatingCount'),
    tf.feature_column.numeric_column('userAvgReleaseYear'),
    tf.feature_column.numeric_column('userReleaseYearStddev'),
    tf.feature_column.numeric_column('userAvgRating'),
    tf.feature_column.numeric_column('userRatingStddev'),
]

preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)

model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid'),
])

model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)

model.fit(train_dataset, epochs=8)

test_loss, test_accuracy = model.evaluate(test_dataset)
print('\n\nTest Lost {}, Test Accuracy {}'.format(test_loss, test_accuracy))
