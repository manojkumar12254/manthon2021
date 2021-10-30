import time
import logging
import os
import h5py
import numpy as np
import json
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import BatchNormalization
from keras.layers import Activation
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.preprocessing import image
from annoy import AnnoyIndex
from keras import optimizers
from keras.models import Model

#initializing the logger 
logger = logging.getLogger()
logger.setLevel(logging.INFO)

def load_headless_pretrained_model():
    #Loads the VGG pretrained model
    print ("Loading the pretrained VGG model...")
    pretrained_vgg16 = VGG16(weights='imagenet', include_top=True)
    model = Model(inputs=pretrained_vgg16.input, outputs=pretrained_vgg16.get_layer('fc2').output)
    return model

def get_feature_vector(model,img_path):
    images = np.zeros(shape=(1, 224, 224, 3))
    img = image.load_img(img_path, target_size=(224, 224))
    x_raw = image.img_to_array(img)
    x_expand = np.expand_dims(x_raw, axis=0)
    images[0, :, :, :] = x_expand
    inputs = preprocess_input(images)
    image_features = model.predict(inputs)
    return image_features[0] 

def generate_features(image_paths, model):
    #Takes in an array of image paths, and a trained model.
    #Returns the activations of the final layer for each image
    print ("Generating the image features...")
    start = time.time()
    images = np.zeros(shape=(len(image_paths), 224, 224, 3))
    file_mapping = {i: f for i, f in enumerate(image_paths)}
    #loading the dataset
    for i, f in enumerate(image_paths):
        img = image.load_img(f, target_size=(224, 224))
        x_raw = image.img_to_array(img)
        x_expand = np.expand_dims(x_raw, axis=0)
        images[i, :, :, :] = x_expand

    logger.info("%s images which are loaded" % len(images))
    inputs = preprocess_input(images)
    logger.info("Images preprocessed")
    images_features = model.predict(inputs)
    end = time.time()
    logger.info("Inference done, %s Generation time" % (end - start))
    return images_features, file_mapping


def save_features(features_filename, features, mapping_filename, file_mapping):
    #Saving the features of image and image mapping
    print ("Saving the image features...")
    np.save('%s.npy' % features_filename, features)
    with open('%s.json' % mapping_filename, 'w') as index_file:
        json.dump(file_mapping, index_file)
    logger.info("Weights saved")


def load_features(features_filename, mapping_filename):
    #Loads features and file_item mapping 
    print ("Loading features...")
    images_features = np.load('%s.npy' % features_filename)
    with open('%s.json' % mapping_filename) as f:
        index_str = json.load(f)
        file_index = {int(k): str(v) for k, v in index_str.items()}
    return images_features, file_index


def index_features(features, n_trees=1000, dims=4096, is_dict=False):
    #Use Annoy to index the features of the images, that able to query them
    print ("Indexing the features of the dataset...")
    feature_index = AnnoyIndex(dims, metric='angular')
    for i, row in enumerate(features):
        vec = row
        if is_dict:
            vec = features[row]
        feature_index.add_item(i, vec)
    feature_index.build(n_trees)
    return feature_index


def build_word_index(word_vectors):
    #Building a fast index out of a list of pretrained word vectors
    print ("Building the word index for the engine...")
    logging.info("Creating mapping and list of features")
    word_list = [(i, word) for i, word in enumerate(word_vectors)]
    word_mapping = {k: v for k, v in word_list}
    word_features = [word_vectors[lis[1]] for lis in word_list]
    logging.info("Building tree")
    word_index = index_features(word_features, n_trees=20, dims=300)
    logging.info("Tree built")
    return word_index, word_mapping


def search_index_by_key(key, feature_index, item_mapping, top_n=10):
    #Search an Annoy index by key, return the nearest datas using the ANN algorthim
    distances = feature_index.get_nns_by_item(key, top_n, include_distances=True)
    return [[a, item_mapping[a], distances[1][i]] for i, a in enumerate(distances[0])]


def search_index_by_value(vector, feature_index, item_mapping, top_n=10):
    #Search an Annoy index by value, return n nearest data using the nearest neighbours algorthim
    distances = feature_index.get_nns_by_vector(vector, top_n, include_distances=True)
    return [[a, item_mapping[a], distances[1][i]] for i, a in enumerate(distances[0])]


def get_weighted_features(class_index, images_features):
    #Use class weights to re-weight our image features and index
    class_weights = get_class_weights_from_vgg()
    target_class_weights = class_weights[:, class_index]
    weighted = images_features * target_class_weights
    return weighted


def get_class_weights_from_vgg(save_weights=False, filename='class_weights'):
    model_weights_path = os.path.join(os.environ.get('HOME'),
                                      '.keras/models/vgg16_weights_tf_dim_ordering_tf_kernels.h5')
    weights_file = h5py.File(model_weights_path, 'r')
    weights_file.get('predictions').get('predictions_W_1:0')
    final_weights = weights_file.get('predictions').get('predictions_W_1:0')

    class_weights = np.array(final_weights)[:]
    weights_file.close()
    if save_weights:
        np.save('%s.npy' % filename, class_weights)
    return class_weights


def setup_custom_model(intermediate_dim=2000, word_embedding_dim=300):
    #Builds a custom model taking the fc2 layer of VGG16 and adding two dense layers on top
    print ("Setting up custom imagenet architecture model ...")
    headless_pretrained_vgg16 = VGG16(weights='imagenet', include_top=True, input_shape=(224, 224, 3))
    x = headless_pretrained_vgg16.get_layer('fc2').output
    for layer in headless_pretrained_vgg16.layers:
        layer.trainable = False

    image_dense1 = Dense(intermediate_dim, name="image_denselayer1")(x)
    image_dense1 = BatchNormalization()(image_dense1)
    image_dense1 = Activation("relu")(image_dense1)
    image_dense1 = Dropout(0.5)(image_dense1)
    image_dense2 = Dense(word_embedding_dim, name="image_denselayer2")(image_dense1)
    image_output = BatchNormalization()(image_dense2)
    complete_model = Model(inputs=[headless_pretrained_vgg16.input], outputs=image_output)
    complete_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return complete_model


def load_glove_vectors(glove_dir, glove_name='glove.6B.300d.txt'):
    glove_emb = open(os.path.join(glove_dir, glove_name))
    embeddings_index = {}
    for line in glove_emb:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings_index[word] = coefs
    glove_emb.close()
    print('Found %s word vectors.' % len(embeddings_index))
    return embeddings_index
