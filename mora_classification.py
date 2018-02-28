import iterator.iterator_clips as it
import models.c3d_model as c3d
from keras import utils as np_utils
import processing.save_video as sv
import numpy as np
import processing.utils as pu
import models.cae3d as cae3d
import models.mora_module as mora
import os
from iterator.data_generator import DataGenerator
import tensorflow as tf
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import pprint

#this module exposes a basic application of MORA execution. Follow the steps to perform the classification using MORA approach

#defining path to read dataset videos
skig_rgb_path = '/Datasets/SKIG_data/RGB/'
skig_of_path = '/Datasets/SKIG_data/Optical_flow/'
skig_depth_path = '/Datasets/SKIG_data/Depth/'

#you could provide an specific weight for each loss function of each gesture class
#remember, the lenght of this loss_weights list should be equal to the number of classes
loss_weights = []

#just adding some examples. Tests conducted on MORA employed a [0.33 0.33 0.33] for every class
loss_weights.append([0.33, 0.33, 0.33])
loss_weights.append([0.3, 0.6, 0.1])
loss_weights.append([0.33, 0.33, 0.33])
loss_weights.append([0.6, 0.2, 0.2])
loss_weights.append([0.1, 0.7, 0.2])
loss_weights.append([0.4, 0.4, 0.2])
loss_weights.append([0.3, 0.6, 0.1])
loss_weights.append([0.3, 0.1, 0.6])
loss_weights.append([0.3, 0.33, 0.33])
loss_weights.append([0.4, 0.5, 0.1])

#number of epochs can also be different for each model. In this example, a single value is used but a list could be provided
number_of_epochs = 180

if __name__ == "__main__":
    
    #list of SKIG folders. An image is provided to show how data is arranged into SKIG folders
    total_list = []
    
    #each entry corresponds to the name of a folder
    gesture_list = ['gesture1', 'gesture2', 'gesture3','gesture4','gesture5','gesture6', 'gesture7', 'gesture8', 'gesture9', 'gesture10']
    
    #iterating over list of gestures
    for index in range (0, len(gesture_list)):
        
        #attributing the correct loss weights and generating a model
        mora_model = mora.create_mora_model(loss_weights[index])
        print("---- Model assembled ! ----") #summary is also being printed. If you dont want to print it, comment this option inside mora module file
        
        #determines the number of splits on the dataset
        number_of_splits = 3;  #train/test/validation
        
        for current_split in range(0, number_of_splits):

            #creates an structure containing all video files from an specific class considering 3 modalities: rgb, optical flow and depth
            #videos are read and subsampled to 32 frames
            rgb_sequences, flow_sequences, depth_sequences, num_samples = it.assembly_video_dataset((gesture_list[index]), skig_rgb_path, skig_of_path, skig_depth_path, current_split, number_of_splits)
            print("---- Training structures assembled ----")

            #each sequence must be split into 4 subsequences (clips) with 8 frames. Each entry has a shape of 4x8x112x112x3
            timestep_size = 8 #each timestep contains 8 frames
            rgb_8frame = it.split_each_video(rgb_sequences, timestep_size)
            flow_8frame = it.split_each_video(flow_sequences, timestep_size)
            depth_8frame = it.split_each_video(depth_sequences, timestep_size)
            print("---- Training videos split into 8-frame clips ----")
            
            #fitting model
            mora.mora_fit(mora_model, rgb_8frame, flow_8frame, depth_8frame, number_of_epochs)
            print("---- Model trained ----")
            
            #Saving model. Remember that each model represents an autoencoder for an specific gesture of an specific split.
            mora_utils.save_weights(mora_model,gesture_list[index])
            print("---- Saving model ----")
        