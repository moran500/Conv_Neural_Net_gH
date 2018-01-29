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
# toto vlastne zmensi obrazok a vytiahne z neho najvacsie hodnoty, na zmensenie pouzijeme maticu 2x2 co je
# zadefinovane argumentom pool_size (2,2)
classifier.add(MaxPooling2D(pool_size = (2, 2)))


#=============================================================================================================