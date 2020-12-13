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

movie_genre_col     = tf.feature_column.categorical_column_with_vocabulary_list(key='movieGenre1', vocabulary_list=genre_vocab)
movie_genre_emb_col = tf.feature_column.embedding_column(movie_genre_col, 10)
movie_genre_ind_col = tf.feature_column.indicator_column(movie_genre_col)

user_genre_col      = tf.feature_column.categorical_column_with_vocabulary_list(key='userGenre1', vocabulary_list=genre_vocab)
user_genre_emb_col  = tf.feature_column.embedding_column(user_genre_col, 10)
user_genre_ind_col  = tf.feature_column.indicator_column(user_genre_col)

movie_col           = tf.feature_column.categorical_column_with_identity(key='movieId', num_buckets=193610)
movie_emb_col       = tf.feature_column.embedding_column(movie_col, 10)
movie_ind_col       = tf.feature_column.indicator_column(movie_col)


user_col = tf.feature_column.categorical_column_with_identity(key='userId', num_buckets=611)
user_emb_col = tf.feature_column.embedding_column(user_col, 10)
user_ind_col       = tf.feature_column.indicator_column(user_col)


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


inputs = {
    'movieGenre1': tf.keras.layers.Input(name='movieGenre1', shape=(), dtype='string'),
    'movieGenre2': tf.keras.layers.Input(name='movieGenre2', shape=(), dtype='string'),
    'movieGenre3': tf.keras.layers.Input(name='movieGenre3', shape=(), dtype='string'),
    'userGenre1': tf.keras.layers.Input(name='userGenre1', shape=(), dtype='string'),
    'userGenre2': tf.keras.layers.Input(name='userGenre2', shape=(), dtype='string'),
    'userGenre3': tf.keras.layers.Input(name='userGenre3', shape=(), dtype='string'),
    'userGenre4': tf.keras.layers.Input(name='userGenre4', shape=(), dtype='string'),
    'userGenre5': tf.keras.layers.Input(name='userGenre5', shape=(), dtype='string'),

    'movieId': tf.keras.layers.Input(name='movieId', shape=(), dtype='int32'),
    'userId': tf.keras.layers.Input(name='userId', shape=(), dtype='int32'),
    'userRatedMovie1': tf.keras.layers.Input(name='userRatedMovie1', shape=(), dtype='int32'),

    'releaseYear': tf.keras.layers.Input(name='releaseYear', shape=(), dtype='int32'),
    'movieRatingCount': tf.keras.layers.Input(name='movieRatingCount', shape=(), dtype='int32'),
    'movieAvgRating': tf.keras.layers.Input(name='movieAvgRating', shape=(), dtype='float32'),
    'movieRatingStddev': tf.keras.layers.Input(name='movieRatingStddev', shape=(), dtype='float32'),
    'userRatingCount': tf.keras.layers.Input(name='userRatingCount', shape=(), dtype='int32'),
    'userAvgReleaseYear': tf.keras.layers.Input(name='userAvgReleaseYear', shape=(), dtype='int32'),
    'userReleaseYearStddev': tf.keras.layers.Input(name='userReleaseYearStddev', shape=(), dtype='float32'),
    'userAvgRating': tf.keras.layers.Input(name='userAvgRating', shape=(), dtype='float32'), 
    'userRatingStddev': tf.keras.layers.Input(name='userRatingStddev', shape=(), dtype='float32'),
}


# 原始特征层，不经过 Embedding，直接用于在输出层连接
orig_feature_columns = [movie_genre_ind_col, user_genre_ind_col, movie_ind_col, user_ind_col]
orig_feature_layer = tf.keras.layers.DenseFeatures(orig_feature_columns)(inputs)


# FM 层，对不同类型的特征进行交叉
movie_genre_emb_layer = tf.keras.layers.DenseFeatures([movie_genre_emb_col])(inputs)
user_genre_emb_layer  = tf.keras.layers.DenseFeatures([user_genre_emb_col])(inputs)
movie_emb_layer       = tf.keras.layers.DenseFeatures([movie_emb_col])(inputs)
user_emb_layer        = tf.keras.layers.DenseFeatures([user_emb_col])(inputs)

movie__user__cross_layer             = tf.keras.layers.Dot(axes=1)([movie_emb_layer, user_emb_layer])
movie__user_genre__cross_layer       = tf.keras.layers.Dot(axes=1)([movie_emb_layer, user_genre_emb_layer])
movie_genre__user__cross_layer       = tf.keras.layers.Dot(axes=1)([movie_genre_emb_layer, user_emb_layer])
movie_genre__user_genre__cross_layer = tf.keras.layers.Dot(axes=1)([movie_genre_emb_layer, user_genre_emb_layer])


# Deep 层
deep_columns = [movie_genre_emb_col, user_genre_emb_col, movie_emb_col, user_emb_col] + numeric_columns
deep_layer = tf.keras.layers.DenseFeatures(deep_columns)(inputs)
deep_layer = tf.keras.layers.Dense(128, activation='relu')(deep_layer)
deep_layer = tf.keras.layers.Dense(128, activation='relu')(deep_layer)


# 连接层
concat_layer = tf.keras.layers.concatenate([orig_feature_layer, movie__user__cross_layer, movie__user_genre__cross_layer,
    movie_genre__user__cross_layer, movie_genre__user_genre__cross_layer, deep_layer])
output_layer = tf.keras.layers.Dense(1, activation='sigmoid')(concat_layer)


model = tf.keras.Model(inputs, output_layer)
model.compile(
    loss='binary_crossentropy',
    optimizer='adam',
    metrics=['accuracy']
)
model.fit(train_dataset, epochs=8, verbose=2)

test_loss, test_accuracy = model.evaluate(test_dataset)
print('\n\nTest Lost {}, Test Accuracy {}'.format(test_loss, test_accuracy))
