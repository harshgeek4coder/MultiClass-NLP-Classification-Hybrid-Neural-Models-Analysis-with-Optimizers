tf.keras.backend.clear_session()
model11 = Sequential()

model11.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model11.add(Dropout(0.5))
model11.add(Bidirectional(LSTM(embedding_dim)))
model11.add(Dense(5, activation='softmax'))

model11.summary()

model11.compile(
    loss='categorical_crossentropy',
    optimizer='adam',
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model11.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t11 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t11)
hist11=history



plot_graphs(hist11),loss_graph(hist11),acc_graph(hist11)

predicted = model11.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))





#with Adagrad
tf.keras.backend.clear_session()
model12 = Sequential()

model12.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model12.add(Dropout(0.5))
model12.add(Bidirectional(LSTM(embedding_dim)))
model12.add(Dense(5, activation='softmax'))

model12.summary()
optim=tf.keras.optimizers.Adagrad(
    learning_rate=0.01
)
model12.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model12.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t12 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t12)
hist12=history

plot_graphs(hist12),loss_graph(hist12),acc_graph(hist12)

# Now we make predictions using the test data to see how the model performs

predicted = model12.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#with SGD
tf.keras.backend.clear_session()
model13 = Sequential()

model13.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model13.add(Dropout(0.5))
model13.add(Bidirectional(LSTM(embedding_dim)))
model13.add(Dense(5, activation='softmax'))

model13.summary()
optim=tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.5, nesterov=True, name="SGD"
)
model13.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model13.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t13 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t13)
hist13=history

plot_graphs(hist13),loss_graph(hist13),acc_graph(hist13)

# Now we make predictions using the test data to see how the model performs

predicted = model13.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#with RMSProp
tf.keras.backend.clear_session()
model14 = Sequential()

model14.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model14.add(Dropout(0.5))
model14.add(Bidirectional(LSTM(embedding_dim)))
model14.add(Dense(5, activation='softmax'))

model14.summary()
optim=tf.keras.optimizers.RMSprop(
    learning_rate=0.01
)
model14.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model14.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t14 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t14)
hist14=history

plot_graphs(hist14),loss_graph(hist14),acc_graph(hist14)

# Now we make predictions using the test data to see how the model performs

predicted = model14.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

#with Adadelta
tf.keras.backend.clear_session()
model15 = Sequential()

model15.add(Embedding(vocab_size, embedding_dim,input_length=train_padded.shape[1]))
model15.add(Dropout(0.5))
model15.add(Bidirectional(LSTM(embedding_dim)))
model15.add(Dense(5, activation='softmax'))

model15.summary()
optim=tf.keras.optimizers.Adadelta(
    learning_rate=0.01
)
model15.compile(
    loss='categorical_crossentropy',
    optimizer=optim,
    metrics=['accuracy'],
)

epochs = 150
batch_size = 32
filepath="weights_best_bi_lstms.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model15.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t15 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t15)
hist15=history

plot_graphs(hist15),loss_graph(hist15),acc_graph(hist15)

# Now we make predictions using the test data to see how the model performs

predicted = model15.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))