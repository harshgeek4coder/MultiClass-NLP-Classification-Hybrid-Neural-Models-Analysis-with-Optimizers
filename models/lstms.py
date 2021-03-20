#With Adam
tf.keras.backend.clear_session()
model6 = Sequential()

model6.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model6.add(Dropout(0.5))
model6.add(LSTM(embedding_dim))
model6.add(Dense(5, activation='softmax'))

model6.summary()

model6.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model6.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t6 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t6)
hist6=history

plot_graphs(hist6),loss_graph(hist6),acc_graph(hist6)

predicted = model6.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))



#With Adagrad :
tf.keras.backend.clear_session()
model7 = Sequential()

model7.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model7.add(Dropout(0.5))
model7.add(LSTM(embedding_dim))
model7.add(Dense(5, activation='softmax'))

model7.summary()
optim=tf.keras.optimizers.Adagrad(
    learning_rate=0.01
)
model7.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model7.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t7 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t7)
hist7=history

plot_graphs(hist7),loss_graph(hist7),acc_graph(hist7)

# Now we make predictions using the test data to see how the model performs

predicted = model7.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With SGD :
tf.keras.backend.clear_session()
model8 = Sequential()

model8.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model8.add(Dropout(0.5))
model8.add(LSTM(embedding_dim))
model8.add(Dense(5, activation='softmax'))

model8.summary()
optim=tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.5, nesterov=True, name="SGD"
)
model8.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model8.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t8 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t8)
hist8=history

plot_graphs(hist8),loss_graph(hist8),acc_graph(hist8)

# Now we make predictions using the test data to see how the model performs

predicted = model8.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With RMSProp :
tf.keras.backend.clear_session()
model9 = Sequential()

model9.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model9.add(Dropout(0.5))
model9.add(LSTM(embedding_dim))
model9.add(Dense(5, activation='softmax'))

model9.summary()
optim=tf.keras.optimizers.RMSprop(
    learning_rate=0.01
)
model9.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model9.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t9 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t9)
hist9=history

plot_graphs(hist9),loss_graph(hist9),acc_graph(hist9)

# Now we make predictions using the test data to see how the model performs

predicted = model9.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#With Adadelta :
tf.keras.backend.clear_session()
model10 = Sequential()

model10.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model10.add(Dropout(0.5))
model10.add(LSTM(embedding_dim))
model10.add(Dense(5, activation='softmax'))

model10.summary()
optim=tf.keras.optimizers.Adadelta(
    learning_rate=0.01)
model10.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model10.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t10 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t10)
hist10=history

plot_graphs(hist10),loss_graph(hist10),acc_graph(hist10)

# Now we make predictions using the test data to see how the model performs

predicted = model10.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))