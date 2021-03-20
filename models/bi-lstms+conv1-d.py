#Bi_directional_LSTMs + CNN1D :
#With Adam
tf.keras.backend.clear_session()
model21 = Sequential()

model21.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model21.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model21.add(MaxPooling1D(pool_size=2))
model21.add(Bidirectional(LSTM(embedding_dim)))
model21.add(Dense(5, activation='softmax'))

model21.summary()

model21.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model21.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t21 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t21)
hist21=history

plot_graphs(hist21),loss_graph(hist21),acc_graph(hist21)

predicted = model21.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))



#With AdaGrad
tf.keras.backend.clear_session()
model22 = Sequential()

model22.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model22.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model22.add(MaxPooling1D(pool_size=2))
model22.add(Bidirectional(LSTM(embedding_dim)))
model22.add(Dense(5, activation='softmax'))

model22.summary()
optim=tf.keras.optimizers.Adagrad(
    learning_rate=0.01
)
model22.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model22.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t22 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t22)
hist22=history

plot_graphs(hist22),loss_graph(hist22),acc_graph(hist22)

# Now we make predictions using the test data to see how the model performs

predicted = model22.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With SGD
tf.keras.backend.clear_session()
model23 = Sequential()

model23.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model23.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model23.add(MaxPooling1D(pool_size=2))
model23.add(Bidirectional(LSTM(embedding_dim)))
model23.add(Dense(5, activation='softmax'))

model23.summary()
optim=tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.5, nesterov=True, name="SGD"
)
model23.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model23.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t23 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t23)
hist23=history

plot_graphs(hist23),loss_graph(hist23),acc_graph(hist23)

# Now we make predictions using the test data to see how the model performs

predicted = model23.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With RMSProp
tf.keras.backend.clear_session()
model24 = Sequential()

model24.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model24.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model24.add(MaxPooling1D(pool_size=2))
model24.add(Bidirectional(LSTM(embedding_dim)))
model24.add(Dense(5, activation='softmax'))

model24.summary()
optim=tf.keras.optimizers.RMSprop(
    learning_rate=0.01
)
model24.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model24.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t24 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t24)
hist24=history

plot_graphs(hist24),loss_graph(hist24),acc_graph(hist24)

# Now we make predictions using the test data to see how the model performs

predicted = model24.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With Adadelta
tf.keras.backend.clear_session()
model25 = Sequential()

model25.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model25.add(Conv1D(filters=32, kernel_size=3, padding='same', activation='relu'))
model25.add(MaxPooling1D(pool_size=2))
model25.add(Bidirectional(LSTM(embedding_dim)))
model25.add(Dense(5, activation='softmax'))

model25.summary()
optim=tf.keras.optimizers.Adadelta(
    learning_rate=0.01
)
model25.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms_n_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model25.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t25 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t25)
hist25=history

plot_graphs(hist25),loss_graph(hist25),acc_graph(hist25)

# Now we make predictions using the test data to see how the model performs

predicted = model25.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))