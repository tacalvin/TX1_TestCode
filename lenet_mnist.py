from cnn.networks.lenet import lenet
from sklearn.cross_validation import train_test_split
from sklearn import datasets
from keras.optimizers import SGD
from keras.utils import np_utils
import numpy as np
import argparse
import cv2
from keras.callbacks import Callback
class LossHistory(Callback):
    def on_train_begin(self, log = {}):
        self.losses = []

    def on_batch_end(self, batch, log = {}):
        self.losses.append(log.get('loss'))
ap = argparse.ArgumentParser()

ap.add_argument("-s", "--save-model", type=int, default=-1,
	help="(optional) whether or not model should be saved to disk")
ap.add_argument("-l", "--load-model", type=int, default=-1,
	help="(optional) whether or not pre-trained model should be loaded")
ap.add_argument("-w", "--weights", type=str,
	help="(optional) path to weights file")
args = vars(ap.parse_args())

print("[DOWNLOADING MNISTY]")
dataset = datasets.fetch_mldata("MNIST Original")

#reshape the mnsist dataset flattened
data = dataset.data.reshape((dataset.data.shape[0],28,28))
data = data[:,np.newaxis,:,:]
(train_data,test_data,train_labels,test_labels) = train_test_split(data / 255.0, dataset.target.astype("int"),test_size= 0.33)

train_labels = np_utils.to_categorical(train_labels,10)
test_labels = np_utils.to_categorical(test_labels,10)

#initalize the optimizer and model
print("[INFO] Compiling model")
opt = SGD(lr = 0.01)
model = lenet.build(width = 28, height = 28,depth = 1,classes = 10,
                    weightPath = args["weights"] if args["load_model"] > 0 else None)
history = LossHistory()
model.compile(loss="categorical_crossentropy",optimizer = opt,metrics=["accuracy"],callbacks=history)

#only train if we are not loading a model
if args["load_model"] < 0:
    print("[INFO] training ...")
    model.fit(train_data,train_labels,batch_size = 128,nb_epoch=20,verbose=1)

    #validate model
    print("[INFO] evaluating...")
    (loss,accuracy) = model.evaluate(test_data,test_labels, batch_size = 128,verbose =1)
    print("[INFO] accuracy: {:.2f}%".format(accuracy*100))

# check to see if the model should be saved to file
print(args.keys())
if args["save_model"] > 0:
	print("[INFO] dumping weights to file...")
	model.save_weights(args["weights"], overwrite=True)
print(history.losses)
#choose random test subjects
for i in np.random.choice(np.arange(0,len(test_labels)), size = (10,)):
    # classify the digit
    probs = model.predict(test_data[np.newaxis,i])
    prediction = probs.argmax(axis =1)

    image = (test_data[i][0] * 255).astype("uint8")
    image = cv2.merge([image] * 3)
    image = cv2.resize(image, (96, 96), interpolation = cv2.INTER_LINEAR)
    cv2.putText(image,str(prediction[0]),(5,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0,255,0),2)
    print("[INFO] Predcicted: {}, Actual: {}".format(prediction[0],np.argmax(test_labels[i])))
    cv2.imshow("Digits",image)
    cv2.waitKey(0)
