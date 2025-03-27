# -*- coding: utf-8 -*-
"""

@author: Tristan DESROUSSEAUX, Gauthier HORVILLE, Charles PRETET, Willy ZHENG
"""

#%%
import os

#Creation of the folders if they're not already created
if not os.path.exists('Training'):      
    os.mkdir('Training')
if not os.path.exists('Testing'):    
    os.mkdir('Testing')

#Adding the features folders if they're not already added
features = ["Five_o_Clock_Shadow","Arched_Eyebrows","Attractive","Bags_Under_Eyes","Bald","Bangs","Big_Lips","Big_Nose","Black_Hair","Blond_Hair","Blurry","Brown_Hair","Bushy_Eyebrows","Chubby","Double_Chin","Eyeglasses","Goatee","Gray_Hair","Heavy_Makeup","High_Cheekbones","Male","Mouth_Slightly_Open","Mustache","Narrow_Eyes","No_Beard","Oval_Face","Pale_Skin","Pointy_Nose","Receding_Hairline","Rosy_Cheeks","Sideburns","Smiling","Straight_Hair","Wavy_Hair","Wearing_Earrings","Wearing_Hat","Wearing_Lipstick","Wearing_Necklace","Wearing_Necktie","Young"]
    
for i in features:
    #Creation of a folder for each attributes
    if not os.path.exists('Training/'+i):
        os.mkdir('Training/'+i)
    if not os.path.exists('Training/'+i):
        os.mkdir('Training/'+i)
    if not os.path.exists('Testing/'+i):
        os.mkdir('Testing/'+i)
    if not os.path.exists('Testing/'+i):
        os.mkdir('Testing/'+i)
    #Creation of a folder if they've got the attribute or not
    if not os.path.exists('Training/'+i+'/Presence_of_feature'):
        os.mkdir('Training/'+i+'/Presence_of_feature')
    if not os.path.exists('Training/'+i+'/Absence_of_feature'):
        os.mkdir('Training/'+i+'/Absence_of_feature')
    if not os.path.exists('Testing/'+i+'/Presence_of_feature'):
        os.mkdir('Testing/'+i+'/Presence_of_feature')
    if not os.path.exists('Testing/'+i+'/Absence_of_feature'):
        os.mkdir('Testing/'+i+'/Absence_of_feature')
    

#%%

#Splitting the data into training(80%) and testing(20%)
import pandas as pd
dataset=pd.read_csv("archive/list_attr_celeba.csv")
dataset = dataset.rename(columns={"5_o_Clock_Shadow": "Five_o_Clock_Shadow"})   #We change the value because they'll be some problems with the 5

training_data = dataset.iloc[:int(202599*0.8),:]
testing_data = dataset.iloc[int(202599*0.8):,:]
   
#%%
import shutil

 
x=0
y=40*202599
for j in features:
    #Training 
    
    for i in training_data.query(j+'==1').iloc[:,:].values:    # Add the photo in the right folder with the right features       
        if not os.path.exists('Training/'+j+'Presence_of_feature/'+i[0]):   #Check if the image is already here or not 
            shutil.copy('archive/img_align_celeba/img_align_celeba/'+i[0],'Training/'+j+'/Presence_of_feature/')
        x=x+1
        print(j+" Training Presence : "+str(x)+'/'+str(y))
            
    for i in training_data.query(j+'==-1').iloc[:,:].values:    # Add the photo in the right folder with the right features       
        if not os.path.exists('Training/'+j+'/Absence_of_feature/'+i[0]):   #Check if the image is already here or not 
            shutil.copy('archive/img_align_celeba/img_align_celeba/'+i[0],'Training/'+j+'/Absence_of_feature/')
        x=x+1
        print(j+" Training Absence : "+str(x)+'/'+str(y))
            
    #Testing 
    for i in testing_data.query(j+'==1').iloc[:,:].values:    # Add the photo in the right folder with the right features       
        if not os.path.exists('Testing/'+j+'/Presence_of_feature/'+i[0]):   #Check if the image is already here or not 
            shutil.copy('archive/img_align_celeba/img_align_celeba/'+i[0],'Testing/'+j+'/Presence_of_feature/')
        x=x+1
        print(j+" Testing Presence : "+str(x)+'/'+str(y))
            
    for i in testing_data.query(j+'==-1').iloc[:,:].values:    # Add the photo in the right folder with the right features       
        if not os.path.exists('Testing/'+j+'Absence_of_feature/'+i[0]):   #Check if the image is already here or not 
            shutil.copy('archive/img_align_celeba/img_align_celeba/'+i[0],'Testing/'+j+'/Absence_of_feature/')
        x=x+1
        print(j+" Testing Absence : "+str(x)+'/'+str(y))
    
#%%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import Dropout

#Initilisation of the CNN
print("CNN Smiling")
classifier=Sequential()

#Step1- Convolution
classifier.add(Conv2D(filters=10, kernel_size=(5,5), activation='relu', input_shape=(64,64,3)))
# 64,64 because of the size of the image 
# 3 because it's RGB 

#Step2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
# 2x2 because of the norme 

##Add a second convolution layer same as the first one 
classifier.add(Conv2D(filters=10, kernel_size=(5,5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step3- Flattening
classifier.add(Flatten())

#Step4 - Full Connection
classifier.add(Dense(units=128, activation='relu')) #Hidden layer with 128 neurons
classifier.add(Dense(units=1, activation='sigmoid') )#Output layer 

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()

# Explication for smiling feature 
# Layer ( type )                  Output Shape               Param # 
#  conv2d_985 (Conv2D)            (None, 60, 60, 10)         760
## Convolution layer              ( batch size , shape of the image, number of filter ) number of parameters
# max_pooling2d_8 (MaxPooling )   ( None,30,30,10 )          0
## Max pooling                    ( batch size, size of the image ( 30 because the images are divided into 2 with the max pooling), filter )
# conv2d_9 ( Conv2D )             ( None,26,26,10 )          2,510
## Same because it's the 2nd layer
# max_pooling2d_9 (MaxPooling )   ( None,13,13,10 )          0
# flatten_4 ( Flatten )           ( None,1690 )              0
## Flattening the images          ( batch size, 13x13x10= 1690 )
# dense_8 ( Dense )               ( None,128 )               216,448 
## Input layer                    ( batch size, number of units )   1690x128 + 128 = 216,448 parameters
# dense_9 ( Dense )               ( None ,1 )                129
## Output layer                   ( batch size, number of unit)   128+1 = 129

train_datagen=ImageDataGenerator( rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)


training_set=train_datagen.flow_from_directory(
    'Training/Smiling',
    target_size=(64,64), #size of images
    batch_size=32,
    class_mode='binary'
)

test_set=test_datagen.flow_from_directory(
    'Testing/Smiling',
    target_size=(64,64), #size of images
    batch_size=32,
    class_mode='binary')


classifier.fit(training_set,
               steps_per_epoch=int(((len(os.listdir("Training\Smiling\Absence_of_feature"))+len(os.listdir("Training\Smiling\Presence_of_feature")))/32)), # number of element in training 
               epochs=5,
               validation_data=test_set,
               validation_steps=int(((len(os.listdir("Testing\Smiling\Absence_of_feature"))+len(os.listdir("Testing\Smiling\Presence_of_feature")))/32))) # number of element in testing 


classifier.save("Smiling.h5")
    
import matplotlib.pyplot as plt

history_smiling = classifier.history.history  # historique de l'entraÃ®nement pour l'attribut 'Smiling'
acc_smiling = history_smiling['accuracy']
val_acc_smiling = history_smiling['val_accuracy']

epochs = range(1, len(acc_smiling) + 1)  # epochs

plt.figure(figsize=(10, 6))

# Courbes pour l'attribut 'Smiling'
plt.plot(epochs, acc_smiling, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc_smiling, 'r', label='Validation Accuracy')

plt.title('Smiling')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

#%%
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
#import Dropout

#Initilisation of the CNN
print("CNN Young")
classifier=Sequential()

#Step1- Convolution
classifier.add(Conv2D(filters=8, kernel_size=(5,5), activation='relu', input_shape=(64,64,3)))
# 64,64 because of the size of the image 
# 3 because it's RGB 

#Step2 - Pooling
classifier.add(MaxPooling2D(pool_size=(2,2)))
# 2x2 because of the norme 

##Add a second convolution layer same as the first one 
classifier.add(Conv2D(filters=8, kernel_size=(5,5), activation='relu'))
classifier.add(MaxPooling2D(pool_size=(2,2)))

# Step3- Flattening
classifier.add(Flatten())

#Step4 - Full Connection
classifier.add(Dense(units=128, activation='relu')) #Hidden layer with 128 neurons
classifier.add(Dense(units=1, activation='sigmoid') )#Output layer 

classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

classifier.summary()

# Explication for smiling feature 
# Layer ( type )                  Output Shape               Param # 
#  conv2d_985 (Conv2D)            (None, 60, 60, 8)          608
## Convolution layer              ( batch size , shape of the image, number of filter ) number of parameters
# max_pooling2d_8 (MaxPooling )   ( None,30,30,8 )           0
## Max pooling                    ( batch size, size of the image ( 30 because the images are divided into 2 with the max pooling), filter )
# conv2d_9 ( Conv2D )             ( None,26,26,8 )           1,608
## Same because it's the 2nd layer
# max_pooling2d_9 (MaxPooling )   ( None,13,13,8 )           0
# flatten_4 ( Flatten )           ( None,1352 )              0
## Flattening the images          ( batch size, 13x13x8= 1352 )
# dense_8 ( Dense )               ( None,128 )               173,184 
## Input layer                    ( batch size, number of units )   1352x128 + 128 = 173,184 parameters
# dense_9 ( Dense )               ( None ,1 )                129
## Output layer                   ( batch size, number of unit)   128+1 = 129

train_datagen=ImageDataGenerator( rescale=1./255,
                                 shear_range=0.2,
                                 zoom_range=0.2,
                                 horizontal_flip=True)

test_datagen=ImageDataGenerator(rescale=1./255)


training_set=train_datagen.flow_from_directory(
    'Training/Young',
    target_size=(64,64), #size of images
    batch_size=32,
    class_mode='binary'
)

test_set=test_datagen.flow_from_directory(
    'Testing/Young',
    target_size=(64,64), #size of images
    batch_size=32,
    class_mode='binary')


classifier.fit(training_set,
               steps_per_epoch=int(((len(os.listdir("Training\Young\Absence_of_feature"))+len(os.listdir("Training\Young\Presence_of_feature")))/32)), # number of element in training 
               epochs=5,
               validation_data=test_set,
               validation_steps=int(((len(os.listdir("Testing\Young\Absence_of_feature"))+len(os.listdir("Testing\Young\Presence_of_feature")))/32))) # number of element in testing 


classifier.save("Young.h5")
    
import matplotlib.pyplot as plt

history_young = classifier.history.history  # historique de l'entraÃ®nement pour l'attribut 'Young'
acc_young = history_young['accuracy']
val_acc_young = history_young['val_accuracy']

epochs = range(1, len(acc_young) + 1)  # epochs

plt.figure(figsize=(10, 6))

# Courbes pour l'attribut 'Young'
plt.plot(epochs, acc_young, 'b', label='Training Accuracy')
plt.plot(epochs, val_acc_young, 'r', label='Validation Accuracy')

plt.title('Young')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)
plt.show()

#%%

# Testing with one image 
from keras.models import load_model
import numpy as np
from keras.preprocessing import image


classifier_cnn=load_model('Smiling.h5')

test_image=image.load_img('archive/img_align_celeba/img_align_celeba/000010.jpg',
                          target_size=(64,64))
test_image=image.img_to_array(test_image).astype('float32')/255
test_image=np.expand_dims(test_image,axis=0)

from sklearn.metrics import classification_report
preds = classifier_cnn.predict(test_image)
print(preds)
training_set.class_indices

if preds>=0.5:
  prediction="Smile"
else:
  prediction='No Smile'

print(prediction)



#%%

#Testing multiple images
from tensorflow.keras.preprocessing import image
#Use the model to predict class of test set
import os

def load_test_data(x,y):
    X_test_input = []
    for i in range(x,y):
        if i<10:
            zero="00000"
        elif (10<=i<100):
            zero="0000"
        elif (100<=i<1000):
            zero="000"
        elif (1000<=i<10000):
            zero="00"    
        img = image.load_img('archive/img_align_celeba/img_align_celeba/'+zero+str(i)+'.jpg',
                                  target_size=(64,64))
        img=image.img_to_array(img).astype('float32')/255
        img=np.expand_dims(img,axis=0)
        X_test_input.append(img)
    return np.concatenate(X_test_input, axis=0 )

X_test_input=load_test_data(10,89)
Y_pred=classifier_cnn.predict(X_test_input)
Y_perc=Y_pred
Y_pred = Y_pred>=0.5

print(Y_pred)

#%%
#import the dataset
dataset=pd.read_csv('archive/list_attr_celeba.csv')
Y_test=dataset.iloc[:, 32].values[9:88]

for i in range (len(Y_test)):
    if Y_test[i] == -1:
        Y_test[i]=0

from sklearn.metrics import confusion_matrix
print(Y_test)
print(Y_pred)
cm=confusion_matrix(Y_pred,Y_test)

print(cm)
accuracy = (cm.diagonal().sum()) / cm.sum()

print("Accuracy:", accuracy)

#%%
import matplotlib.pyplot as plt
import numpy as np

def plot_image(i, predictions_array, true_label, img): 
    predictions_array, true_label, img = predictions_array[i], true_label[i], img[i]
    plt.grid(False) 
    plt.xticks([]) 
    plt.yticks([]) 
    plt.imshow(img, cmap=plt.cm.binary) 
    predicted_label = np.argmax(predictions_array)
    if predicted_label == true_label: 
        color = 'blue' 
    else: 
        color = 'red' 
    val = ["Smiling","No Smiling"]
    plt.xlabel("{} {:2.0f}% ({})".format(val[predicted_label], 100*np.max(predictions_array), val[true_label]), color=color)
        
       
def plot_value_array(i, predictions_array, true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.xticks([])
    plt.yticks([])
    
    # DÃ©finir les couleurs pour chaque classe (bleu pour la classe 1 et rouge pour la classe 0)
    colors = ['red', 'blue']
    
    # CrÃ©er le graphique Ã  barres en spÃ©cifiant les couleurs pour chaque classe
    thisplot = plt.bar(range(2), predictions_array, color=[colors[label] for label in range(2)])
    
    plt.ylim([0, 1])
    predicted_label = np.argmax(predictions_array)
    
    thisplot[predicted_label].set_height(predictions_array[0])  
    thisplot[true_label].set_height(1 - predictions_array[0])

i = 0
plt.figure(figsize=(12, 6))  
plt.subplot(1, 2, 1)  
plot_image(i, Y_perc, Y_test, X_test_input)  
plt.subplot(1, 2, 2)  
plot_value_array(i, Y_perc, Y_test)  
plt.show()

#%%
def plot_images_with_predictions(images, predictions, true_labels):
    plt.figure(figsize=(15, 5))
    for i in range(len(images)):
        plt.subplot(2, 5, i + 1)
        plt.xticks([])
        plt.yticks([])
        plt.grid(False)
        plt.imshow(images[i], cmap=plt.cm.binary)
        predicted_label = "Smile" if predictions[i] else "No Smile"
        true_label = "Smile" if true_labels[i] else "No Smile"
        color = 'blue' if predicted_label == true_label else 'red'
        plt.xlabel(f"Predicted: {predicted_label}\nTrue: {true_label}", color=color)
    plt.show()

# Utilisation :
plot_images_with_predictions(X_test_input[10:20], Y_pred[10:20], Y_test[10:20])

#%%
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow import keras


def create_model(filters=6):
    classifier = Sequential()
    classifier.add(Conv2D(filters=filters, kernel_size=(5,5), activation='relu', input_shape=(64,64,3)))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(filters=filters, kernel_size=(5,5), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid') )
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


filters=[5,6,7,8,9,10]
model = KerasClassifier(model=create_model,filters=filters, batch_size=10,epochs=2)
param_grid = dict(filters=filters)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=10,n_jobs=1)

Y_train=dataset.iloc[:, 32].values[0:5065]
for i in range (len(Y_train)):
    if Y_train[i] == -1:
        Y_train[i]=0


X_train=load_test_data(1,5066)

grid.fit(X_train, Y_train) 

print(grid.best_params_) 
grid_predictions = grid.predict(X_test_input) 
   
# print classification report 
print(classification_report(Y_test, grid_predictions)) 

#sumarize results 
grid_result=grid.fit(X_train, Y_train)
print("Best for Smiling: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#Best : 0.862780 using {'filters': 10}
#On a fait un GridSearch pour connaÃ®tre le meilleur filtre et on a trouvÃ© 10 avec une prÃ©cision de 0.86


#%%
from sklearn.model_selection import GridSearchCV
from scikeras.wrappers import KerasClassifier, KerasRegressor
from tensorflow import keras


def create_model(filters=7):
    classifier = Sequential()
    classifier.add(Conv2D(filters=filters, kernel_size=(5,5), activation='relu', input_shape=(64,64,3)))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Conv2D(filters=filters, kernel_size=(5,5), activation='relu'))
    classifier.add(MaxPooling2D(pool_size=(2,2)))
    classifier.add(Flatten())
    classifier.add(Dense(units=128, activation='relu'))
    classifier.add(Dense(units=1, activation='sigmoid') )
    classifier.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return classifier


filters=[5,6,7,8,9,10]
model = KerasClassifier(model=create_model,filters=filters, batch_size=10,epochs=2)
param_grid = dict(filters=filters)

grid = GridSearchCV(estimator=model, param_grid=param_grid, scoring='accuracy', cv=10,n_jobs=1)

Y_train=dataset.iloc[:, 40].values[0:5065]
for i in range (len(Y_train)):
    if Y_train[i] == -1:
        Y_train[i]=0


X_train=load_test_data(1,5066)

grid.fit(X_train, Y_train) 

print(grid.best_params_) 
grid_predictions = grid.predict(X_test_input) 
   
# print classification report 
print(classification_report(Y_test, grid_predictions)) 

#sumarize results 
grid_result=grid.fit(X_train, Y_train)
print("Best for Young: %f using %s" % (grid_result.best_score_, grid_result.best_params_))

#Best for Young: 0.817964 using {'filters': 8}
#On a fait un GridSearch pour connaÃ®tre le meilleur filtre et on a trouvÃ© 10 avec une prÃ©cision de 0.81

#%%

import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing import image

Smile_Young=0
Smile_Old =0
NotSmile_Young =0
NotSmile_Old = 0

for i in range(10,99):
    #l'image
    classifier_cnn1=load_model('Smiling.h5')
    classifier_cnn2=load_model('Young.h5')

    test_image=image.load_img('archive/img_align_celeba/img_align_celeba/0000'+str(i)+'.jpg',
                             target_size=(64,64))
    test_image=image.img_to_array(test_image).astype('float32')/255
    test_image=np.expand_dims(test_image,axis=0)
    #les preds de l'image
    preds_Smiling = classifier_cnn1.predict(test_image)
    preds_Young = classifier_cnn2.predict(test_image)
    if preds_Smiling>=0.5 :
        if preds_Young>=0.5:
            Smile_Young=Smile_Young+1
        else:
            Smile_Old=Smile_Old+1
    else:
        if preds_Young>=0.5:
             NotSmile_Young=NotSmile_Young+1
        else:
             NotSmile_Old=NotSmile_Old+1



#Cree graph
plt.figure(figsize=(10, 6))

plt.bar(0, Smile_Young, color='blue', label='Nb personnes qui sourient et qui sont jeunes')
plt.bar(1, Smile_Old, color='orange', label='Nb personnes qui sourient et qui ne sont pas jeunes')
plt.bar(2, NotSmile_Young, color='red', label='Nb personnes qui ne sourient pas et qui sont jeunes')
plt.bar(3, NotSmile_Old, color='green', label='Nb personnes qui ne sourient pas  et qui ne sont pas jeunes')

plt.ylabel('Valeur')
plt.title("Comparaison des distributions selon l'IA")
plt.xticks([]) 
plt.legend()

plt.show()
#%%
import matplotlib.pyplot as plt
Y_test_Smiling=dataset.iloc[:, 32].values[10:99]
Y_test_Young=dataset.iloc[:,40].values[10:99]

Smile_Young=0
Smile_Old =0
NotSmile_Young =0
NotSmile_Old = 0

for i in range (len(Y_test_Smiling)):
    if Y_test_Smiling[i]==1 :
        if Y_test_Young[i]==1:
            Smile_Young= Smile_Young+1
        else:
            Smile_Old =Smile_Old +1
    else:
         if Y_test_Young[i]==1:
             NotSmile_Young= NotSmile_Young+1
         else:
             NotSmile_Old =NotSmile_Old +1


#DÃ©finition des donnÃ©es
data1 = 3
data2 = 4
data3 = 5
data = 6

#CrÃ©ation du graphique en barres
plt.figure(figsize=(10, 6))

plt.bar(0, Smile_Young, color='blue', label='Nb personnes qui sourient et qui sont jeunes')
plt.bar(1, Smile_Old, color='orange', label='Nb personnes qui sourient et ne sont pas jeunes')
plt.bar(2, NotSmile_Young, color='red', label='Nb personnes qui ne sourient pas et sont jeunes')
plt.bar(3, NotSmile_Old, color='green', label='Nb personnes qui ne sourient pas et ne sont pas jeunes')

plt.ylabel('Valeur')
plt.title('Comparaison des distributions selon la base de donnÃ©es')
plt.xticks([]) 
plt.legend()

plt.show()