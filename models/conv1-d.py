#With Adam
tf.keras.backend.clear_session()
model = Sequential()
model.add(Embedding(vocab_size, embedding_dim, input_length=train_padded.shape[1]))

model.add(Conv1D(48, 5, activation='relu', padding='valid'))
model.add(GlobalMaxPooling1D())
model.add(Dropout(0.5))

model.add(Flatten())
model.add(Dropout(0.5))

model.add(Dense(5, activation='softmax'))
model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

epochs = 150
batch_size = 32
filepath="weights_best_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t1 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t1)
hist1=history





plot_graphs(hist1),loss_graph(hist1),acc_graph(hist1)











# Now we make predictions using the test data to see how the model performs

predicted = model.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

# Testing Random Text and Predicted Classes :

original_text=raw_df['Text']
text = original_text[18]
new_text = [clean_text(text)]
print(text)
print(new_text)
seq = tokenizer.texts_to_sequences(new_text)
padded = pad_sequences(seq, maxlen=max_length, padding=padding_type, truncating=trunc_type)
pred = model.predict(padded)
acc = model.predict_proba(padded)
predicted_label = encode.inverse_transform(pred)
print('')
print(f'Product category id: {np.argmax(pred[0])}')
print(f'Predicted label is: {predicted_label[0]}')
print(f'Accuracy score: { acc.max() * 100}')





# With AdaGrad
tf.keras.backend.clear_session()
model2 = Sequential()
model2.add(Embedding(vocab_size, embedding_dim, input_length=train_padded.shape[1]))

model2.add(Conv1D(48, 5, activation='relu', padding='valid'))
model2.add(GlobalMaxPooling1D())
model2.add(Dropout(0.5))

model2.add(Flatten())
model2.add(Dropout(0.5))

model2.add(Dense(5, activation='softmax'))

optim=tf.keras.optimizers.Adagrad(
    learning_rate=0.01
    
)
model2.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

epochs = 150
batch_size = 32
filepath="weights_best_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model2.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t2 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t2)
hist2=history


plot_graphs(hist2),loss_graph(hist2),acc_graph(hist2)

# Now we make predictions using the test data to see how the model performs

predicted = model2.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

# With SGD
tf.keras.backend.clear_session()
model3 = Sequential()
model3.add(Embedding(vocab_size, embedding_dim, input_length=train_padded.shape[1]))

model3.add(Conv1D(48, 5, activation='relu', padding='valid'))
model3.add(GlobalMaxPooling1D())
model3.add(Dropout(0.5))

model3.add(Flatten())
model3.add(Dropout(0.5))

model3.add(Dense(5, activation='softmax'))

optim=tf.keras.optimizers.SGD(
    learning_rate=0.01, momentum=0.5, nesterov=True, name="SGD"
)
model3.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

epochs = 150
batch_size = 32
filepath="weights_best_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model3.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t3 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t3)
hist3=history


plot_graphs(hist3),loss_graph(hist3),acc_graph(hist3)

# Now we make predictions using the test data to see how the model performs

predicted = model3.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

# With RMSPROP
tf.keras.backend.clear_session()
model4 = Sequential()
model4.add(Embedding(vocab_size, embedding_dim, input_length=train_padded.shape[1]))

model4.add(Conv1D(48, 5, activation='relu', padding='valid'))
model4.add(GlobalMaxPooling1D())
model4.add(Dropout(0.5))

model4.add(Flatten())
model4.add(Dropout(0.5))

model4.add(Dense(5, activation='softmax'))

optim=tf.keras.optimizers.RMSprop(
    learning_rate=0.01
    
)
model4.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

epochs = 150
batch_size = 32
filepath="weights_best_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model4.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t4 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t4)
hist4=history


plot_graphs(hist4),loss_graph(hist4),acc_graph(hist4)

# Now we make predictions using the test data to see how the model performs

predicted = model4.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))

# With Adadelta
tf.keras.backend.clear_session()

model5 = Sequential()
model5.add(Embedding(vocab_size, embedding_dim, input_length=train_padded.shape[1]))

model5.add(Conv1D(48, 5, activation='relu', padding='valid'))
model5.add(GlobalMaxPooling1D())
model5.add(Dropout(0.5))

model5.add(Flatten())
model5.add(Dropout(0.5))

model5.add(Dense(5, activation='softmax'))

optim=tf.keras.optimizers.Adadelta(
    learning_rate=0.01
)
model5.compile(loss='categorical_crossentropy', optimizer=optim, metrics=['accuracy'])

epochs = 150
batch_size = 32
filepath="weights_best_cnn.hdf5"

checkpoint = ModelCheckpoint(filepath, monitor='val_accuracy', verbose=1, save_best_only=True, mode='max',save_weights_only=True)
start = time.perf_counter()
callbacks_list = [checkpoint,EarlyStopping(monitor='val_accuracy', mode='max', patience=10, verbose=1)]


history = model5.fit(train_padded, training_labels, shuffle=True ,
                    epochs=epochs, batch_size=batch_size, 
                    callbacks=callbacks_list,validation_data=(validation_padded, validation_labels))
t5 = time.perf_counter() - start
print('Total time took for training %.3f seconds.' % t5)
hist5=history


plot_graphs(hist5),loss_graph(hist5),acc_graph(hist5)

# Now we make predictions using the test data to see how the model performs

predicted = model5.predict(validation_padded)
evaluate_preds(np.argmax(validation_labels, axis=1), np.argmax(predicted, axis=1))