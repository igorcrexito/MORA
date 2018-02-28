import cv2
import numpy as np
import os
import pprint

resized_width = 112
resized_height = 112
clip_number = 8
minimum_number_of_frames = 32

#this method reads a dataset folder and its subfolder gathering all videos, augmenting and returning in a list
def assembly_video_dataset(gesture_list, rgb_video_path, of_video_path, depth_video_path, offset, number_of_splits):
    
    #list to store videos and optical flow sequences
    rgb_inputs = []
    flow_inputs = []
    depth_inputs = []
    
    for subject_index in range (0,len(gesture_list)):
    
        #for each gesture, add their videos to the list
        subject = gesture_list[subject_index]
        listOfVideos = os.listdir(rgb_video_path + subject)
        samples_per_iteration = len(listOfVideos)/number_of_splits
        
        counter = offset*samples_per_iteration
        for video in range (0,len(listOfVideos)):
            
            if counter >= (offset+1)*samples_per_iteration:
                break
            
            video_name = listOfVideos[int(counter)]
            if video_name[-4:] != ".avi":
                continue
            
            #load the original video
            cap = cv2.VideoCapture(rgb_video_path + subject + '/' + video_name)

            #load flow video
            flow_cap = cv2.VideoCapture(of_video_path + subject + '/' + video_name)
            
            #load depth videos
            depth_video = 'K' + video_name[1:]
            depth_cap = cv2.VideoCapture(depth_video_path + subject + '/' + depth_video)

            #get frame by frame and store in a list
            rgb = []
            
            #get flow frames and store in a list
            flow = []
            
            #get depth information and store in a list
            depth = []
            
            
            #getting the number of frames of the videoo
            number_of_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if number_of_frames >= minimum_number_of_frames:
                #calculating frame interval to get 32 frames per video (sampling)
                video_clip_frames = minimum_number_of_frames
                frame_interval = int(number_of_frames / video_clip_frames)

                for i in range (0, number_of_frames):
                    ret, img = cap.read()

                    #get flow frames
                    ret, flow_img = flow_cap.read()

                    #get depth frames
                    ret, depth_img = depth_cap.read()

                    #add the last frame
                    if not ret:
                        #store rgb frames on arrays
                        common_frame = cv2.resize(img, (resized_width, resized_height))
                        rgb.append(common_frame)

                        #store flows on arrays
                        common_frame = cv2.resize(flow_img, (resized_width, resized_height))
                        flow.append(common_frame)

                        #store depths on arrays
                        common_frame = cv2.resize(depth_img, (resized_width, resized_height))
                        depth.append(common_frame)

                        common_frame = cv2.resize(depth_img_rotated, (resized_width, resized_height))
                        rotated_depth.append(common_frame)

                        break

                    if i % frame_interval == 0:
                        common_frame = cv2.resize(img, (resized_width, resized_height))
                        rgb.append(common_frame)

                        common_frame = cv2.resize(flow_img, (resized_width, resized_height))
                        flow.append(common_frame)

                        common_frame = cv2.resize(depth_img, (resized_width, resized_height))
                        depth.append(common_frame)


                #sample 32-frame clips from each transformed video and their flows
                start_frame = 0
                X = rgb
                X = rgb[start_frame:(start_frame + video_clip_frames)]  
                flowX = flow
                flowX = flow[start_frame:(start_frame + video_clip_frames)]  
                depthX = depth
                depthX = depth[start_frame:(start_frame + video_clip_frames)]  

                #append each video and to the list
                for i in range (0, len(X)):
                    X[i] = X[i] / 255
                    flowX[i] = flowX[i] / 255
                    depthX[i] = depthX[i] / 255

                #appending to list
                rgb_inputs.append(X)
                flow_inputs.append(flowX)
                depth_inputs.append(depthX)

                counter = counter + 1

    return rgb_inputs, flow_inputs, depth_inputs, len(rgb_inputs)


#this method produces 8-frame clips
def split_each_video(training_input, frame_interval):
    
    #get the number of video samples
    number_of_samples = len(training_input)
    
    new_training_samples = []
    
    #split each video into 8-frame clips and append to the list
    for index in range (0, number_of_samples):
        video = training_input[index]
        
        max_index = 0
        min_index = 0
        counter = 0
        while (max_index < len(video)):
            min_index = counter * frame_interval
            max_index = (counter + 1) * frame_interval
            new_training_samples.append(video[min_index:max_index])
            counter = counter + 1
        
    return new_training_samples
