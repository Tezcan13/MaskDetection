import numpy as np
import os
import matplotlib.pyplot as plt
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Flatten
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing.image import img_to_array
from tensorflow.keras.preprocessing.image import load_img
from tensorflow.keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

directory= r"C:\Users\Tezcn\Desktop\imagelist" #Directory where our imagelist folder is located
classes = ["mask", "unmask"] #We have 2 classes, mask and unmask

imagelist = [] #We will use this for appending our images
labellist = [] #This is for appending our categories, mask and unmask

for category in classes:
    path = os.path.join(directory, category)
    for img in os.listdir(path): #os.listdir returns a list containing the names of the entries in the directory given by path
    	img_path = os.path.join(path, img) #join the path with corresponding image
    	image = load_img(img_path, target_size=(224, 224)) #It takes images and set the target size 224*224 as we defined, we can change it but it is the stable value 
    	image = img_to_array(image) #We convert images to array
    	image = preprocess_input(image) #To use MobileNetV2, we need this

    	imagelist.append(image) #We will add our images to the list
    	labellist.append(category) #We will add categories to the list

#Categorize mask and unmask using 0,1 . Our images converted in array, we will also use numbers for mask and unmask categories
lb = LabelBinarizer()
labellist = lb.fit_transform(labellist)
labellist = to_categorical(labellist)

#Converting to numpy arrays
imagelist = np.array(imagelist)
labellist = np.array(labellist)

#Partition the data into training and testing splits using 80% of the data for training and the remaining 20% for testing
(trainX, testX, trainY, testY) = train_test_split(imagelist, labellist, test_size=0.20, stratify=labellist, random_state=42)

#Imagelist augmentation with the help of informations from a website (https://keras.io/api/preprocessing/image/#imagedatagenerator-class)
aug = ImageDataGenerator(
	rotation_range=20, #Integer, degree range for random rotations
	zoom_range=0.15, #Float, range for random zoom
	width_shift_range=0.2, #Float, fraction of total width
	height_shift_range=0.2, #Float, fraction of total height
	shear_range=0.15, #Float, shear intensity
	horizontal_flip=True, #Boolean value for randomly slipping horizontally
	fill_mode="nearest") #Choose between {constant, nearest, reflect, wrap}. Default is nearest

#Load the MobileNetV2 network, there are some pretrained images in imagenet, it will give better results 
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224, 224, 3)))

#headModel will be our output model, baseModel was our input model
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7, 7))(headModel) #Average pooling calculates the average value for each patch on the feature map
headModel = Flatten(name="flatten")(headModel) #Flattening is converting the data into a 1-D array
headModel = Dense(128, activation="relu")(headModel) #Relu is activation function for nonlinear use case, we use this for input images
headModel = Dropout(0.5)(headModel) #Dropout is a regularization technique, it deletes some connections
headModel = Dense(2, activation="softmax")(headModel) #This is for output, we have 2 outputs, with and without mask, we use softmax for output images

#This will become the actual model we will train
model = Model(inputs=baseModel.input, outputs=headModel)

#With the help of this loop, layers will not change during training
for layer in baseModel.layers:
	layer.trainable = False

LEARNING_RATE = 0.0001 #Determines the step size at each iteration, it is generally 0.0001
EPOCHS = 25 #It shows the model I am training will train the entire imagelistset 25 times, if we increase this number we get better accuracy but it takes much time
BS = 32 #It doesn't help us to get a good result, it is a set of samples used in one iteration for training, generally 16 or 32


#Compiling the model using Adam optimizer
opt = Adam(lr=LEARNING_RATE, decay=LEARNING_RATE / EPOCHS)
model.compile(loss="binary_crossentropy", optimizer=opt, metrics=["accuracy"])

#Train the head of the network
H = model.fit(
	aug.flow(trainX, trainY, batch_size=BS),
	steps_per_epoch=len(trainX) // BS,
	validation_data=(testX, testY),
	validation_steps=len(testX) // BS,
	epochs=EPOCHS)

#Make predictions
predIdxs = model.predict(testX, batch_size=BS)

#For each image, we need to find the index of the label with max prediction
predIdxs = np.argmax(predIdxs, axis=1)

#Save model
model.save("mask.model", save_format="h5")

#If you dont need classification report and accuracy graph, you can delete all the next lines

#Classification report
print(classification_report(testY.argmax(axis=1), predIdxs, target_names=lb.classes_))

#Graph for the training loss and accuracy
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Accuracy Graph")
plt.xlabel("Epoch")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")