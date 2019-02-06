def create_embedding_graph(input_vec_size, output_count, output_size, embedding_dim=300):
    inputs = tf.keras.Input(shape=(input_vec_size,))
    embedding = tf.keras.layers.Dense(embedding_dim, activation='relu')(inputs)
    outputs = []
    for i in range(0, output_count):
        outputs.append(tf.layers.Dense(output_size, activation='softmax')(embedding))
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=tf.train.AdamOptimizer(0.01),
                  loss='mse',
                  metrics=['mse'])

    return model
