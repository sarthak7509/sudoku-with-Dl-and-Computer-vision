# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 
#   *Copyrights are in name of :- SARTHAK BHATNAGAR/Python_is_pie   #
#                                                                   #
# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # 

from sudoko import SudokoNet
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.datasets import mnist
from sklearn.preprocessing import LabelBinarizer
from sklearn.metrics import classification_report
import argparse


INIT_LR = 1e-3
EPOCHS = 10
BS=128

print("[info] Loading mnist Datasets....")

((trainData,trainLabels),(testData,testLabels)) = mnist.load_data()

# add a channel (i.e., grayscale) dimension to the digits
trainData = trainData.reshape((trainData.shape[0], 28, 28, 1))
testData = testData.reshape((testData.shape[0], 28, 28, 1))


trainData = trainData.astype("float32") / 255.0
testData = testData.astype("float32") / 255.0

le = LabelBinarizer()
trainLabels = le.fit_transform(trainLabels)
testLabels = le.transform(testLabels)

print("[info] compiling model...")
opt = Adam(lr=INIT_LR)
model = SudokoNet.build(width=28,height=28,depth=1,classes=10)
model.compile(loss="categorical_crossentropy",optimizer=opt,metrics=["accuracy"])

print("[info] models training...")
H = model.fit(
	trainData,trainLabels,
	validation_data=(testData,testLabels),batch_size=BS,epochs=EPOCHS,verbose=1)

print("[info] evaluating model predictions...")
predictions = model.predict(testData)
print(classification_report(
	testLabels.argmax(axis=1),
	predictions.argmax(axis=1),
	target_names=[str(x) for x in le.classes_]))

print("[info] saving our models")
model.save("digits_model.h5",save_format="h5")
