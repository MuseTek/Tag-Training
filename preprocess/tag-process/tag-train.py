'''
Classify sounds using database
Author: Scott H. Hawley
This is kind of a mixture of Keun Woo Choi's code https://github.com/keunwoochoi/music-auto_tagging-keras
   and the MNIST classifier at https://github.com/fchollet/keras/blob/master/examples/mnist_cnn.py
Trained using Fraunhofer IDMT's database of monophonic guitar effects,
   clips were 2 seconds long, sampled at 44100 Hz
'''
from __future__ import print_function
import numpy as np
import librosa
from panotti.models import *
from panotti.datautils import *
from keras.callbacks import ModelCheckpoint,EarlyStopping
import os
from os.path import isfile
from timeit import default_timer as timer

def train_network(weights_file="weights.hdf5", classpath="Preproc/Train/", epochs=50, batch_size=20, val_split=0.25,tile=False):
    np.random.seed(1)

    #create the data generator
    train_gen = build_free_sounds_dataset(path='/projects/MuseTek/preprocess/tag-process/Preproc/Train',batch_size=batch_size,tile=tile)
    val_gen = build_free_sounds_dataset(path='/projects/MuseTek/preprocess/tag-process/Preproc/Val',batch_size=batch_size,tile=tile)
    test_gen = build_free_sounds_dataset(path='/projects/MuseTek/preprocess/tag-process/Preproc/Test',batch_size=batch_size,tile=tile)
    X_train_sample, y_train_sample = next(train_gen)

    # Instantiate the model
    model, serial_model = setup_model(X_train_sample, y_train_sample.shape[1], weights_file=weights_file)

    # save_best_only = (val_split > 1e-6)
    # checkpointer = MultiGPUModelCheckpoint(filepath=weights_file, verbose=1, save_best_only=save_best_only,
    #       serial_model=serial_model, period=1, class_names=class_names)
    earlystopping = EarlyStopping(patience=12)

    checkpoint = ModelCheckpoint(weights_file, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint,earlystopping]
    model.fit_generator(train_gen,steps_per_epoch=1000,epochs=100,callbacks=callbacks_list,validation_data=val_gen,validation_steps=100)

    # model.fit(X_train, Y_train, batch_size=batch_size, epochs=epochs, shuffle=True,
    #       verbose=1, callbacks=[checkpointer], validation_split=val_split)  # validation_data=(X_val, Y_val),

    # Score the model against Test dataset
    # X_test, Y_test, paths_test, class_names_test  = build_dataset(path=classpath+"../Test/", tile=tile)
    # assert( class_names == class_names_test )

    score = model.evaluate_generator(test_gen)
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description="trains network using training dataset")
    parser.add_argument('-w', '--weights', #nargs=1, type=argparse.FileType('r'),
        help='weights file in hdf5 format', default="weights.hdf5")
    parser.add_argument('-c', '--classpath', #type=argparse.string,
        help='Train dataset directory with list of classes', default="Preproc/Train/")
    parser.add_argument('--epochs', default=20, type=int, help="Number of iterations to train for")
    parser.add_argument('--batch_size', default=40, type=int, help="Number of clips to send to GPU at once")
    parser.add_argument('--val', default=0.25, type=float, help="Fraction of train to split off for validation")
    parser.add_argument("--tile", help="tile mono spectrograms 3 times for use with imagenet models",action="store_true")
    args = parser.parse_args()
    train_network(weights_file=args.weights, classpath=args.classpath, epochs=args.epochs, batch_size=args.batch_size,
        val_split=args.val, tile=args.tile)