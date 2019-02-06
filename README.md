# Digital Singal Processing Project
This branch is decicated to final term project for digital signal processing course in Shahid Beheshti  University. 

The  report of the project can found in [Project Final Report](./doc/Final/Main.pdf)

# How-to Setup 
Here are the steps to setup  the project from scratch. 
## Download and Process Input Files 
The project is built based on Librispeech project data. 
1. Goto Librispeech site and download the dataset [Librispeech](http://www.openslr.org/12/) 
2. Extract dataset and run bin/import_librivox.py script to convert the files and have 
a master CSV file to some output folder 
3. In DeepSpeechVector.py run the prepare_data method it will create two files logits.csv and 
lgram.txt 
4. In DeepSpeechVector.py run the train_model method to train and test the model

