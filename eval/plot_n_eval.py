def plot_graphs(history):
  plt.title('Loss VS Accuracy')
  plt.plot(history.history['loss'], label='train loss')
  plt.plot(history.history['val_loss'], label='test loss')
  plt.legend()
  
  plt.plot(history.history['accuracy'], label='train accuracy')
  plt.plot(history.history['val_accuracy'], label='test accuracy')
  plt.legend()
  plt.show();
  return None
def loss_graph(history):
  plt.title('Loss')
  plt.plot(history.history['loss'], label='train')
  plt.plot(history.history['val_loss'], label='test')
  plt.legend()
  plt.show();

  return None

def acc_graph(history):
  plt.title('Accuracy')
  plt.plot(history.history['accuracy'], label='train')
  plt.plot(history.history['val_accuracy'], label='test')
  plt.legend()
  plt.show();

  return None


def evaluate_preds(y_true, y_preds):
    """
    Performs evaluation comparison on y_true labels vs. y_pred labels
    on a classification.
    """
    accuracy = accuracy_score(y_true, y_preds)
    precision = precision_score(y_true, y_preds, average='micro')
    recall = recall_score(y_true, y_preds, average='micro')
    f1 = f1_score(y_true, y_preds, average='micro')
    metric_dict = {"accuracy": round(accuracy, 2),
                   "precision": round(precision, 2),
                   "recall": round(recall, 2),
                   "f1": round(f1, 2)}
    print(f"Acc: {accuracy * 100:.2f}%")
    print(f"Precision: {precision:.2f}")
    print(f"Recall: {recall:.2f}")
    print(f"F1 score: {f1:.2f}")
    
    return metric_dict

def model_eval(model,a,b,c,d):
  score=model.evaluate(a,b,verbose=0)
  score1=model.evaluate(c,d,verbose=0)
  print("\n")
  print("Model Loss on training data ",score[0])
  print("Model Accuracy on training data: ",score[1])
  print("Model Loss on validation data",score1[0])
  print("Model Accuracy on validation data: ",score1[1])
  print("\n")
  return score,score1

def model_eval(model,a,b,c,d):
  score=model.evaluate(a,b,verbose=0)
  score1=model.evaluate(c,d,verbose=0)
  print("\n")
  print("Model Loss on training data ",score[0])
  print("Model Accuracy on training data: ",score[1])
  print("Model Loss on validation data",score1[0])
  print("Model Accuracy on validation data: ",score1[1])
  print("\n")
  return score,score1