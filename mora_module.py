from keras.layers import Input, GRU, Lambda, Conv3D, UpSampling3D, MaxPooling3D, Reshape, concatenate
from keras.models import Model

#routine to create a multi-output recurrent autoencoder
#the input parameter is the weight for each different loss function: ex.: loss_weights = [0.33, 0.33, 0.33];

def create_mora_model(weights):
    
    #specifying the input shape (24-frames clips) -> # timesteps, frames, height, width, channels
    #24 frames representing a composed input with 8-frame for each modality
    input = Input(batch_shape=(4,24,112,112,3), name='main_input')
    
    #encoding input into a higher level representation - 1st group
    encoding = Conv3D(24, (3,3,3), activation='relu', padding='same')(input)
    encoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool1')(encoding)
    
    encoding = Conv3D(128, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(128, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool2')(encoding)
    
    encoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(encoding)
    encoding = MaxPooling3D(pool_size=(2, 2, 2), strides=(2, 2, 2),
                           border_mode='valid', name='pool3')(encoding)
    
    encoding = Conv3D(64, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(32, (1,3,3), activation='relu', padding='same')(encoding)
    encoding = Conv3D(12, (1,3,3), activation='relu', padding='same')(encoding)
    
    #reshaping tensor to return with GRU
    reshaped = Reshape((7056,1))(encoding)
    encoder = GRU(392, activation='sigmoid', recurrent_activation='hard_sigmoid', use_bias=True, return_state = True, stateful = False)(reshaped)
    
    #concatenating both states and outputs
    encoder = concatenate([encoder[0], encoder[1]], axis = 1)
    encoder = Reshape((4,14,14,1))(encoder)
    encoding = Reshape((36,14,14,1))(encoding)
    
    #creating a mixed representation with GRU representation and encoded representation for each time step
    concatenateLayer = concatenate([encoding, encoder], axis = 1)
    encoding = Reshape((1,14,14,40))(concatenateLayer)
    
    #decoding step -> getting back to original representation
    decoding = Conv3D(16, (1,3,3), activation='relu', padding='same')(encoding)
    decoding = Conv3D(32, (1,3,3), activation='relu', padding='same')(decoding)
    decoding = Conv3D(64, (1,3,3), activation='relu', padding='same')(decoding)
    decoding = UpSampling3D((3,1,1))(decoding)
    
    decoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(decoding)
    decoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(decoding)
    decoding = UpSampling3D((2,2,2))(decoding)
    
    decoding = Conv3D(128, (3,3,3), activation='relu', padding='same')(decoding)
    decoding = Conv3D(128, (3,3,3), activation='relu', padding='same')(decoding)
    decoding = UpSampling3D((2,2,2))(decoding)
    
    decoding = Conv3D(64, (3,3,3), activation='relu', padding='same')(decoding)
    decoding = Conv3D(8, (3,3,3), activation='relu', padding='same')(decoding)
    decoding = UpSampling3D((2,2,2))(decoding)
    
    #lambda layers to split the representation into 3 portions
    out1 = Lambda(slice_tensor1)(decoding)
    out2 = Lambda(slice_tensor2)(decoding)
    out3 = Lambda(slice_tensor3)(decoding)
    
    #this is the first output of the model -> hand shape
    output1 = Conv3D(3, (3,3,3), activation='sigmoid', padding='same')(out1)
    
    #this is the second output of the model -> motion
    output2 = Conv3D(3, (3,3,3), activation='sigmoid', padding='same')(out2)
    
    #this is the second output of the model -> motion
    output3 = Conv3D(3, (3,3,3), activation='sigmoid', padding='same')(out3)
    
    #autoencoder = Model([input], [output1, output2, output3])
    autoencoder = Model([input], [output1, output2, output3])
    autoencoder.compile(optimizer='adadelta', loss='mean_squared_error', loss_weights = weights)
    autoencoder.summary()
    
    return autoencoder

#methods to slice intermediate representation into 3 different portions
def slice_tensor1(input):
    return input[:,0:8,:,:,:]

def slice_tensor2(input):
    return input[:,8:16,:,:,:]

def slice_tensor3(input):
    return input[:,16:24,:,:,:]

#method to perform the fit on a MORA model
#list of parameters: mora model previously created, 8frame sequences of RGB videos, 8frame sequences of flow videos, 8frame sequences of depth videos, number of training epochs

def mora_fit(model, rgb_sequence, flow_sequence, depth_sequence, number_of_epochs):
    
    #reshaping inputs to be suitable for the model
    rgb_sequence = np.reshape(rgb_sequence, (len(rgb_sequence), 8, 112, 112, 3))
    flow_sequence = np.reshape(flow_sequence, (len(flow_sequence),8, 112, 112, 3))
    depth_sequence = np.reshape(depth_sequence, (len(depth_sequence),8, 112, 112, 3))
    
    #concatenating and creating a 24-frame representation; containing RGB, Flow and depth
    composed_input = np.concatenate((rgb_sequence, flow_sequence), axis=1)
    composed_input = np.concatenate((composed_input, depth_sequence), axis=1)
    
    #model fit -> provide a validation set if you have one
    #batch size must be a multiple of the timestep dimension
    #no shuffle should be performed in order to maintain the time correlation between timesteps
    model.fit(composed_input, [rgb_sequence, flow_sequence, depth_sequence],
                epochs=number_of_epochs,
                batch_size=4,
                shuffle=False,
                validation_data=(composed_input, [rgb_sequence, flow_sequence, depth_sequence]),
                callbacks=[TensorBoard(log_dir='/tmp/autoencoder')])
                
    return model

