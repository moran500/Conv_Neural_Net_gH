# Convolutional Neural networks
#=============================================================================================================
# Building the CNN

# Import the Keras librares and packages

from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense

# Initialising the CNN
classifier = Sequential()

# Convolution layer
# toto pridava convolution layer co vlastne urobi vynasobenie medzi obrazkom a feature detector a vysledkom su jednotlive feature maps
# argument 32 znamena ze budem mat 32 roznych feature detector, co znamena ze budem mat 32 roznych feature maps
# 3,3 argument znamena ze feature detector bude mat 3 riadky a 3 stlpce, cize bude to matica 3x3
# input_shape definuje ake rozlisenie bude mat vstupny obrazok a kolko vrstiev bude mat jedna feature map. ak sa jedna o ciernobileu fotku tak tam bude 1 a ak sa jedna o farebnu tak 3 lebo mam tri vrstvy RED, GREEN, BLUE
# activation znamena ze aktivacna funkcia bude rectifier
classifier.add(Conv2D(32, (3, 3), input_shape = (64, 64, 3), activation = 'relu' ))

# Max Pooling
# toto vlastne vytiahne zo vsetkych features maps ich najvacsie hodnoty a tiez zmensi rozmer kazdej feature map, na toto zmensenie pouzije maticu 2x2 co je
# zadefinovane argumentom pool_size (2,2)
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# co sa tu stane je ze aplikujeme convilation na feature maps ktore sme vytvorili prvou convolution layer
# Tu pridame dalsiu convolution layer aby sme zvysili presnost na testovacich datach, lebo bez tejto vrstvy
# sme mali sice presnost na traning data cez 90% ale na testovacich iba 75% a toto ma priblizit
# presnost testovacich dat k tym trainingovym datam
# keby sme este viac chceli zvacsich presnot treba zvacsit obrazok z 64x64 pixelov na viac cim dodame viac dat
# parameter input_shape mozeme vyhodit lebo tato vrstva nieje prva v poradi cize Keras vie co tam dat
classifier.add(Conv2D(32, (3, 3), activation = 'relu' ))
# Max Pooling
classifier.add(MaxPooling2D(pool_size = (2, 2)))

# Flattening
# toto prehodi vsetky feature maps ktore su po max poolingu na jeden dlhy vektor ktory bude vstupnou vrstvou pre NN
classifier.add(Flatten())

# Connecting the CNN
# toto je to iste ako v minulom priklade takze vytvorime jednu hidden layer s 128 nodmi
# a jednu output layer s 1 vystupom
classifier.add(Dense( units = 128, activation = 'relu'))
classifier.add(Dense( units = 1, activation = 'sigmoid'))

# Compile the C NN
classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy' , metrics = ['accuracy'])

#=============================================================================================================

# Image preprocessing
# tento kod som stiahol z keros documentacie zo sekcie preprocessing a pod sekcie Image
# tato cast vlastne urobi to ze vytvori dalsie obrazku z tych co uz mam a transformuje ich roznym sposobom

from keras.preprocessing.image import ImageDataGenerator

# Tato trieda ImageDataGenerator
# rescale urobi to ze vsetky pixely prehodi na hodnoty medzi 0 a 1 lebo normalne kazdy pixel moze mat hodnotu od 0 do 255
# shear_range ze zase nejaka transformacia a tiez zoom_range je transformacia
# posledny je horizontal_flip co znamena ze nam horizontalne otoci obrazok
# toto je urobene pre training data preto sa premenna vola train_datagen
train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

# toto je zase to iste len pre testovacie data
# tu dame len rescale a nic viac lebo su to testovacie data
test_datagen = ImageDataGenerator(rescale=1./255)

# tato funkcia hovori o tom ze z kade pridu trainingove data a v akom formate a co s nimi
# takze target_size je predpokladany format obrazku, tu zvolime to iste co sme zvolili v Convolation layer
# batch_size je ze sa zabalia obrazky do batcho po 32 obrazkov, to je na backward propagation
# class_mode hovori o tom ci depended variable je binary alebo ma viac moznosti ako len 2
training_set = train_datagen.flow_from_directory(
        'dataset/training_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

test_set = test_datagen.flow_from_directory(
        'dataset/test_set',
        target_size=(64, 64),
        batch_size=32,
        class_mode='binary')

# tato funkcia spusta cele trenovanie a je treba aby sme tam zadali okrem premennych pre training set a premennej za test data
# ale treba zadat steps_per_epoch co je vlastne pocet obrazkov v training sete, validation_steps co je vlastne pocet obrazkov v test sete
# a epochs co je vlastne pocet epoch pri trenovani.
classifier.fit_generator(
        training_set,
        steps_per_epoch=8000,
        epochs=25,
        validation_data = test_set,
        validation_steps=2000)


# tuto si chcem ulozit model ktory som natrenoval aby som ho potom nemusel trenovat znovu a len ho
# importnem z tohoto filu, ten druhy save by mal ulozit len jednotlive vahu modelu.
classifier.save('CNN_model.h5')
classifier.save_weights('CNN_model_weights.h5')

#=============================================================================================================

# home work - treba otestovat ako predikovat obrazok podla nami natrenovanej siete

import numpy as np
from keras.preprocessing import image

# tato funkcia mi naloaduje obrazok ktory chcem odtestovat
# druhy parameter je target_size a to je vlastne rozlisenie v akom chceme mat obrazok
# musi to byt take iste rozlisenie ako je pouzite v modeli
img = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg', target_size = (64, 64))

# tato funkcia mi vytvori z 2 dimenzionalneho obrazku, 3 dimensionalny ale nie ze by bol 3D
# ale znamenato ze prida dimenziu pre farby. Ak je to farebny obrazok tak to budu 3 vrstvy
img_array = image.img_to_array(img)

# tato funkcia prida dalsiu dimensiu, lebo model na vstupe caka premennu so 4 dimensiami
img_array_4_dims = np.expand_dims(img_array, axis = 0)

# tu sa vytvori predikcia
new_prediction = classifier.predict(img_array_4_dims)

# tato funkcia nam ukaze na vystupe ake su jednotlive indexy pre jednotlive vysledky co chceme v nasom pripade:
# {'cats': 0, 'dogs': 1}
training_set.class_indices

# vysledok predikcie je bud 1 alebo 0 a z predchadzajucej funkcie vieme ze 1 je dog a 0 je cat
if new_prediction[0][0] == 1:
    result = 'dog'
else:
    result = 'cat'






#=============================================================================================================



























