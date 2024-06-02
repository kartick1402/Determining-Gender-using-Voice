import os
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard, EarlyStopping # type: ignore

from utils import load_data, split_data, create_model

X, y = load_data()

data = split_data(X, y, test_size=0.1, valid_size=0.1)

model = create_model()

# use tensorboard to view metrics
tensorboard = TensorBoard(log_dir="logs")
early_stopping = EarlyStopping(mode="min", patience=5, restore_best_weights=True)
batch_size = 64
epochs = 100

model.fit(data["X_train"], data["y_train"], epochs=epochs, batch_size=batch_size, validation_data=(data["X_valid"], data["y_valid"]),
          callbacks=[tensorboard, early_stopping])

model.save("results/model.h5")

print("Now after Modelling finally we will be evaluating the model with the help of",len(data['X_test'])," samples...")
loss, accuracy = model.evaluate(data["X_test"], data["y_test"], verbose=0)
print("Loss:", loss)
print("Accuracy: ",accuracy*100,"%")