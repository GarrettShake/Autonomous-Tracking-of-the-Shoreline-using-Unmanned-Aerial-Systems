contours.py should be run first, note the commented sections change depending on the video
folders (2) need to be created before it is run to store images, and they need to be specified in the code

feature_extraction.py creates a .xlsx file used to train the algorithm. specify the folders created from 
contours.py

bhv1.csv is the dataset fo bob_hall.mp4
v2.csv is the dataset for cape_cod_edit.mp4

bh_fin.py is used to run the model for bob_hall.mp4
cc_fin.py is used to run the model for cape_cod_edit.mp4
They both do the same thing for different videos, the accuracy printed is base solely on the dataset, not the
video itself so take it with a grain of salt. make sure the files are specified correctly, and create an 
output folder.