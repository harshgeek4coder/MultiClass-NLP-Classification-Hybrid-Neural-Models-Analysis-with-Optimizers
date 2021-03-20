print("Final Performance Of All Models : \n")
print("Model 1 - CNN-1D \n")





print("With Adam : \n")
mod1=model_eval(model,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaGrad : \n")
mod2=model_eval(model2,train_padded,training_labels,validation_padded,validation_labels)
print("With SGD : \n")
mod3=model_eval(model3,train_padded,training_labels,validation_padded,validation_labels)
print("With RMSProp : \n")
mod4=model_eval(model4,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaDelta : \n")
mod5=model_eval(model5,train_padded,training_labels,validation_padded,validation_labels)

print("Model 2 - LSTM \n")

print("With Adam : \n")
mod6=model_eval(model6,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaGrad : \n")
mod7=model_eval(model7,train_padded,training_labels,validation_padded,validation_labels)
print("With SGD : \n")
mod8=model_eval(model8,train_padded,training_labels,validation_padded,validation_labels)
print("With RMSProp : \n")
mod9=model_eval(model9,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaDelta : \n")
mod10=model_eval(model10,train_padded,training_labels,validation_padded,validation_labels)


print("Model 3 - BiDirectional LSTM \n")
mod11=model_eval(model11,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaGrad : \n")
mod12=model_eval(model12,train_padded,training_labels,validation_padded,validation_labels)
print("With SGD : \n")
mod13=model_eval(model13,train_padded,training_labels,validation_padded,validation_labels)
print("With RMSProp : \n")
mod14=model_eval(model14,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaDelta : \n")
mod15=model_eval(model15,train_padded,training_labels,validation_padded,validation_labels)

print("Model 4 - CNN 1-D + LSTM \n")

print("With Adam : \n")
mod16=model_eval(model16,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaGrad : \n")
mod17=model_eval(model17,train_padded,training_labels,validation_padded,validation_labels)
print("With SGD : \n")
mod18=model_eval(model18,train_padded,training_labels,validation_padded,validation_labels)
print("With RMSProp : \n")
mod19=model_eval(model19,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaDelta : \n")
mod20=model_eval(model20,train_padded,training_labels,validation_padded,validation_labels)

print("Model 5 - CNN 1-D + Bi Directional LSTM \n")

print("With Adam : \n")
mod21=model_eval(model21,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaGrad : \n")
mod22=model_eval(model22,train_padded,training_labels,validation_padded,validation_labels)
print("With SGD : \n")
mod23=model_eval(model23,train_padded,training_labels,validation_padded,validation_labels)
print("With RMSProp : \n")
mod24=model_eval(model24,train_padded,training_labels,validation_padded,validation_labels)
print("With AdaDelta : \n")
mod25=model_eval(model25,train_padded,training_labels,validation_padded,validation_labels)

table = PrettyTable()

table.field_names = ['Model [With Optimiser]', 'Accuracy','Time Taken To Train (secs)']

table.add_row(['CNN-1D [Adam]',round(mod1[1][1]*100,2),round(t1,2)])
table.add_row(['CNN-1D [AdaGrad]', round(mod2[1][1]*100,2),round(t2,2)])
table.add_row(['CNN-1D [SGD]', round(mod3[1][1]*100,2),round(t3,2)])
table.add_row(['CNN-1D [RMSProp]', round(mod4[1][1]*100,2),round(t4,2)])
table.add_row(['CNN-1D [AdaDelta]', round(mod5[1][1]*100,2),round(t5,2)])

table.add_row(['LSTM [Adam]',round(mod6[1][1]*100,2),round(t6,2)])
table.add_row(['LSTM [AdaGrad]', round(mod7[1][1]*100,2),round(t7,2)])
table.add_row(['LSTM [SGD]', round(mod8[1][1]*100,2),round(t8,2)])
table.add_row(['LSTM [RMSProp]', round(mod9[1][1]*100,2),round(t9,2)])
table.add_row(['LSTM [AdaDelta]', round(mod10[1][1]*100,2),round(t10,2)])



table.add_row(['Bidirectional LSTM [Adam]',round(mod11[1][1]*100,2),round(t11,2)])
table.add_row(['Bidirectional LSTM [AdaGrad]', round(mod12[1][1]*100,2),round(t12,2)])
table.add_row(['Bidirectional LSTM [SGD]', round(mod13[1][1]*100,2),round(t13,2)])
table.add_row(['Bidirectional LSTM [RMSProp]', round(mod14[1][1]*100,2),round(t14,2)])
table.add_row(['Bidirectional LSTM [AdaDelta]', round(mod15[1][1]*100,2),round(t15,2)])



table.add_row([' CNN-1D + LSTM  [Adam]',round(mod16[1][1]*100,2),round(t16,2)])
table.add_row([' CNN-1D + LSTM  [AdaGrad]', round(mod17[1][1]*100,2),round(t17,2)])
table.add_row([' CNN-1D + LSTM  [SGD]', round(mod18[1][1]*100,2),round(t18,2)])
table.add_row([' CNN-1D + LSTM  [RMSProp]', round(mod19[1][1]*100,2),round(t19,2)])
table.add_row([' CNN-1D + LSTM  [AdaDelta]', round(mod20[1][1]*100,2),round(t20,2)])



table.add_row([' CNN-1D + Bidirectional LSTM  [Adam]',round(mod21[1][1]*100,2),round(t21,2)])
table.add_row([' CNN-1D + Bidirectional LSTM  [AdaGrad]', round(mod22[1][1]*100,2),round(t22,2)])
table.add_row([' CNN-1D + Bidirectional LSTM  [SGD]', round(mod23[1][1]*100,2),round(t23,2)])
table.add_row([' CNN-1D + Bidirectional LSTM  [RMSProp]', round(mod24[1][1]*100,2),round(t24,2)])
table.add_row([' CNN-1D + Bidirectional LSTM  [AdaDelta]', round(mod25[1][1]*100,2),round(t25,2)])


print(table)

print(table.get_string(sortby='Accuracy',reversesort=True))

print(table.get_string(sortby='Time Taken To Train (secs)',reversesort=True))