import tensorflow as tf


def train_model(x_train, y_train, num_nodes, dropout, batch_size, lr, epochs):
    nn_model = tf.keras.models.Sequential(
        [
            tf.keras.layers.Dense(num_nodes, activation="relu", input_shape=(10,)),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(num_nodes, activation="relu"),
            tf.keras.layers.Dropout(dropout),
            tf.keras.layers.Dense(1, activation="sigmoid"),
        ]
    )

    nn_model.compile(
        optimizer=tf.keras.optimizers.Adam(lr),
        loss=tf.keras.losses.BinaryCrossentropy(),
        metrics=["accuracy"],
    )
    history = nn_model.fit(
        x_train, y_train, batch_size=batch_size, epochs=epochs, validation_split=0.2
    )

    return nn_model, history
