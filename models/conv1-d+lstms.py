#With CNN 1-D + LSTMs
#With Adam
tf.keras.backend.clear_session()
model16 = Sequential()

model16.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model16.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model16.add(MaxPooling1D(pool_size=2))
model16.add(LSTM(embedding_dim))
model16.add(Dense(5, activation='softmax'))

model16.summary()

model16.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model16.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t16 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t16)
hist16=history

plot_graphs(hist16),loss_graph(hist16),acc_graph(hist16)

predicted = model16.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))



#With AdaGrad
tf.keras.backend.clear_session()
model17 = Sequential()

model17.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model17.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model17.add(MaxPooling1D(pool_size=2))
model17.add(LSTM(embedding_dim))
model17.add(Dense(5, activation='softmax'))

model17.summary()
optim=tf.keras.optimizers.Adagrad(
    learning_rate=0.01
)
model17.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model17.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t17 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t17)
hist17=history

plot_graphs(hist17),loss_graph(hist17),acc_graph(hist17)

# Now we make predictions using the test data to see how the model performs

predicted = model17.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With SGD
tf.keras.backend.clear_session()
model18 = Sequential()

model18.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model18.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model18.add(MaxPooling1D(pool_size=2))
model18.add(LSTM(embedding_dim))
model18.add(Dense(5, activation='softmax'))

model18.summary()
optim=tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.5, nesterov=True, name="SGD"
)
model18.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model18.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t18 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t18)
hist18=history

plot_graphs(hist18),loss_graph(hist18),acc_graph(hist18)

# Now we make predictions using the test data to see how the model performs

predicted = model18.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With RMSprop
tf.keras.backend.clear_session()
model19 = Sequential()

model19.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model19.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model19.add(MaxPooling1D(pool_size=2))
model19.add(LSTM(embedding_dim))
model19.add(Dense(5, activation='softmax'))

model19.summary()
optim=tf.keras.optimizers.RMSprop(
    learning_rate=0.01
)
model19.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model19.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t19 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t19)
hist19=history

plot_graphs(hist19),loss_graph(hist19),acc_graph(hist19)

# Now we make predictions using the test data to see how the model performs

predicted = model19.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With AdaDelta
tf.keras.backend.clear_session()
model20 = Sequential()

model20.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model20.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model20.add(MaxPooling1D(pool_size=2))
model20.add(LSTM(embedding_dim))
model20.add(Dense(5, activation='softmax'))

model20.summary()
optim=tf.keras.optimizers.Adadelta(
    learning_rate=0.01
)
model20.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model20.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t20 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t20)
hist20=history

plot_graphs(hist20),loss_graph(hist20),acc_graph(hist20)

# Now we make predictions using the test data to see how the model performs

predicted = model20.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))