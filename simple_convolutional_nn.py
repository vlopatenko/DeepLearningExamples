import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


mnist_data = input_data.read_data_sets("MNIST_data/", one_hot=True)
# Инициализируем вход и выход сети
x = tf.placeholder(dtype=tf.float32, shape=[None, 784])
y = tf.placeholder(dtype=tf.float32, shape=[None, 10])

# Переформатируем входной вектор в виде двумерного массива,
# Первая размерность (-1) - заранее неизвестный объем мини-батча
x_image = tf.reshape(x, shape=[-1, 28, 28, 1])

# Инициализируем веса свертки (32 фильтра размера 5x5) и свободные члены
W_conv_1 = tf.Variable(tf.truncated_normal([5, 5, 1, 32], stddev=0.1))
b_conv_1 = tf.Variable(tf.constant(0.1, shape=[32]))

# Инициализируем сверточный слой
conv_1 = tf.nn.conv2d(x_image, W_conv_1, strides=[1, 1, 1, 1], padding="SAME") + b_conv_1
h_conv_1 = tf.nn.relu(conv_1)

# Определяем слой субдискретизации
h_pool_1 = tf.nn.max_pool(h_conv_1, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Параметры второго сверточного слоя
W_conv_2 = tf.Variable(tf.truncated_normal([5, 5, 32, 64], stddev=0.1))
b_conv_2 = tf.Variable(tf.constant(0.1, shape=[64]))

# Сам сверточный слой #2
conv_2 = tf.nn.conv2d(h_pool_1, W_conv_2, strides=[1, 1, 1, 1], padding="SAME")
h_conv_2 = tf.nn.relu(conv_2)

# Второй слой max pool
h_pool_2 = tf.nn.max_pool(h_conv_2, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

# Из двумерного слоя делаем плоский для применения полносвязного слоя
h_pool_2_flat = tf.reshape(h_pool_2, [-1, 7*7*64])

# Добавляем первый полносвязный слой
W_fc_1 = tf.Variable(tf.truncated_normal([7*7*64, 1024], stddev=0.1))
b_fc_1 = tf.Variable(tf.constant(0.1, shape=[1024]))
h_fc_1 = tf.nn.relu(tf.matmul(h_pool_2_flat, W_fc_1) + b_fc_1)

# Регуляризуем его дропаутом
keep_probability = tf.placeholder(tf.float32)
h_fc_1_drop = tf.nn.dropout(h_fc_1, keep_prob=keep_probability)

# Добавляем последний полносвязный слой с 10 выходами (классы)
W_fc_2 = tf.Variable(tf.truncated_normal([1024, 10], stddev=0.1))
b_fc_2 = tf.Variable(tf.constant(0.1, shape=[10]))
y_conv = tf.nn.softmax(tf.matmul(h_fc_1_drop, W_fc_2) + b_fc_2)

# Определяем функцию ошибки как перекрестную энтропию
cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=y_conv))
train_step = tf.train.AdamOptimizer(0.0001).minimize(cross_entropy)

# Оцениваем точность
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

#Запускаем обучение
init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)
for i in range(10000):
    print(i / 10000 * 100)
    batch_xs, batch_ys = mnist_data.train.next_batch(64)
    sess.run(train_step,
             feed_dict={x: batch_xs, y: batch_ys, keep_probability: 0.5})

print(sess.run(accuracy,
               feed_dict={x: mnist_data.test.images, y: mnist_data.test.labels, keep_probability: 1}))
