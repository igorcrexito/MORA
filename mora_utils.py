from keras.models import model_from_json

def save_weights(model, name):
    model_json = model.to_json()
    with open("skig_" + name + "_classification_.json", "w") as json_file:
        json_file.write(model_json)
    
    # serialize weights to HDF5
    model.save_weights("skig_" + name+"_classification_model.h5")
    print("Saved model to disk")


def load_weights(name):
    json_file = open("skig_"+name + "_classification_.json", 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    loaded_model = model_from_json(loaded_model_json)
    # load weights into new model
    loaded_model.load_weights("skig_"+name + "_classification_model.h5")
    print("Loaded model from disk")
 
    return loaded_model