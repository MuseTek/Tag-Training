from __future__ import print_function
import numpy as np
from panotti.datautils import *
import librosa
from audioread import NoBackendError
import os
from PIL import Image
from functools import partial
from imageio import imwrite
import multiprocessing as mp
from sklearn import preprocessing
import pdb
# from utils.resolve_osx_aliases import resolve_osx_alias

# this is either just the regular shape, or it returns a leading 1 for mono
def get_canonical_shape(signal):
    if len(signal.shape) == 1:
        return (1, signal.shape[0])
    else:
        return signal.shape

def find_max_shape(path, mono=False, sr=None, dur=None, clean=False):
    if (mono) and (sr is not None) and (dur is not None):   # special case for speedy testing
        return [1, int(sr*dur)]
    shapes = []
    for dirname, dirnames, filenames in os.walk(path):
        for filename in filenames:
            if not (filename.startswith('.') or ('.csv' in filename)):    # ignore hidden files & CSVs
                filepath = os.path.join(dirname, filename)
                try:
                    signal, sr = librosa.load(filepath, mono=mono, sr=sr)
                except NoBackendError as e:
                    print("Could not open audio file {}".format(filepath))
                    raise e
                if (clean):                           # Just take the first file and exit
                    return get_canonical_shape(signal)
                shapes.append(get_canonical_shape(signal))

    return (max(s[0] for s in shapes), max(s[1] for s in shapes))

def convert_one_file(printevery, class_files, nb_classes, classname, n_load, dirname, resample, mono, already_split, nosplit, n_train, n_validate, outpath, subdir, max_shape, clean, out_format, mels, phase, file_index):
    infilename = class_files[file_index]
    audio_path = dirname + '/' + infilename

    if (0 == file_index % printevery) or (file_index+1 == len(class_files)):
        print("\r Processing class ",1,"/",nb_classes,": \'",classname,
            "\', File ",file_index+1,"/", n_load,": ",audio_path,"                             ",
            sep="",end="\r")
    sr = None
    if (resample is not None):
        sr = resample
    try:

        signal, sr = load_audio(audio_path, mono=mono, sr=sr)
    except:
        return

    # Reshape / pad so all output files have same shape
    shape = get_canonical_shape(signal)     # either the signal shape or a leading one
    if (shape != signal.shape):             # this only evals to true for mono
        signal = np.reshape(signal, shape)
        #print("...reshaped mono so new shape = ",signal.shape, end="")
    #print(",  max_shape = ",max_shape,end="")
    padded_signal = np.zeros(max_shape)     # (previously found max_shape) allocate a long signal of zeros
    use_shape = list(max_shape[:])
    use_shape[0] = min(shape[0], max_shape[0] )
    use_shape[1] = min(shape[1], max_shape[1] )
    #print(",  use_shape = ",use_shape)
    padded_signal[:use_shape[0], :use_shape[1]] = signal[:use_shape[0], :use_shape[1]]

    layers = make_layered_melgram(padded_signal, sr, mels=mels, phase=phase)

    if not already_split and (not nosplit):
        print("file index: %i " % (file_index, ))
        if (file_index < n_validate):

            outsub = "Validate/"
        elif (file_index >= n_validate and file_index < n_validate + n_train):
            outsub = "Train/"
        else:
            outsub = "Test/"
    elif nosplit:
        outsub = ""
    else:
        outsub = subdir

    outfile = outpath + outsub  + '/' + infilename+'.'+out_format

    # print(outfile)
    save_melgram(outfile, layers, out_format=out_format)
    return


def create_tag_dict(tag_file):

    master_tag_set = set()
    #currently assuming there are no duplicates
    with open(tag_file) as f:
        tag_info = f.readlines()
    tag_info = [x.strip() for x in tag_info]
    for line in tag_info:
        tags = []
        info_list = line.split(',')
        filename = info_list[1]
        for i in range(2,7):
            tags.append(info_list[i])
            if(info_list[i] != ''):
                master_tag_set.add(info_list[i])
    
    master_tag_list = list(master_tag_set)
    le = preprocessing.LabelEncoder()
    le.fit(master_tag_list)

    tag_label_dict = {}

    num_total_tags = len(master_tag_list)
    
    for line in tag_info:
        info_list = line.split(',')
        filename = info_list[1]
        label_np = np.zeros(num_total_tags)
        current_tags = []
        for i in range(2,7):
            if(info_list[i] != ''):
                current_tags.append(info_list[i])
        tag_indexes = le.transform(current_tags)
        label_np[tag_indexes] = 1
        tag_label_dict[filename] = label_np
    return tag_label_dict



def preprocess_dataset(inpath="Samples/", outpath="Preproc/", train_percentage=0.8, validate_percentage=0.1, resample=None, already_split=False,
    nosplit=False, sequential=False, mono=False, dur=None, clean=False, out_format='npy', mels=96, phase=False):

    if (resample is not None):
        print(" Will be resampling at",resample,"Hz",flush=True)
    
    if (True == already_split):
        print(" Data is already split into Train & Test",flush=True)
        sampleset_subdirs = ["Train/", "Test/"]
    elif nosplit:
        print(" All files output to same directory",flush=True)
        sampleset_subdirs = ["./"]
    else:        
        sampleset_subdirs = ["./"]

    
    if (True == sequential):
        print(" Sequential ordering",flush=True)
    else:
        print(" Shuffling ordering",flush=True)
    
    print(" Finding max shape...",flush=True)
    max_shape = find_max_shape(inpath, mono=mono, sr=resample, dur=dur, clean=clean)

    print(''' Padding all files with silence to fit shape:
              Channels : {}
              Samples  : {}
          '''.format(max_shape[0], max_shape[1]))

    if nosplit:
        train_outpath = outpath
        test_outpath = outpath
        validate_outpath = outpath
    else:
        train_outpath = outpath+"Train/"
        test_outpath = outpath+"Test/"
        validate_outpath = outpath+"Validate/"
    if not os.path.exists(outpath):
        os.mkdir( outpath );   # make a new directory for preproc'd files
        if not nosplit:
            os.mkdir( train_outpath )
            os.mkdir( test_outpath )
            os.mkdir( validate_outpath )
    
    else:
        train_outpath = outpath
        test_outpath = outpath
        validate_outpath = outpath
    
    parallel = False     # set to false for debugging. when parallel jobs crash, usually no error messages are given, the system just hangs
    if (parallel):
        cpu_count = os.cpu_count()
        print("",cpu_count,"CPUs detected: Parallel execution across",cpu_count,"CPUs",flush=True)
    else:
        cpu_count = 1
        print("Serial execution",flush=True)
    

    #now go through all the samples and create spectrograms
    if not os.path.exists(train_outpath):
        print("Making directory ",train_outpath)
        os.mkdir(train_outpath)
        if not nosplit:
            os.mkdir(test_outpath)
    dirname = inpath
    
    class_files = list(listdir_nohidden(inpath))   # all filenames for this class, skip hidden files
    class_files.sort()
    if (not sequential): # shuffle directory listing (e.g. to avoid alphabetic order)
        np.random.shuffle(class_files)   # shuffle directory listing (e.g. to avoid alphabetic order)
    n_files = len(class_files)
    n_load = n_files            # sometimes we may multiple by a small # for debugging
    n_train = int( n_load * train_percentage )
    n_validate = int( n_load * validate_percentage )

    printevery = 20             # how often to output status messages when processing lots of files

    file_indices = tuple( range(len(class_files)) )
    class_index = 1
    nb_classes = 1
    classname = 'tagging-set'
    subdir = ''
    if (not parallel):
        print(len(file_indices))
        for file_index in file_indices:    # loop over all files
            convert_one_file(printevery, class_files, nb_classes, classname, n_load, dirname,resample, mono, already_split, nosplit, n_train, n_validate, outpath, subdir, max_shape, clean, out_format, mels, phase, file_index)
    else:
        pool = mp.Pool(cpu_count)
        pool.map(partial(convert_one_file, printevery, class_files, nb_classes, classname, n_load, dirname,resample, mono, already_split, nosplit, n_train, n_validate, outpath, subdir, max_shape, clean, out_format, mels, phase), file_indices)
        pool.close() # shut down the pool

    print("")    # at the very end, newline
    return

if __name__ == '__main__':
    import platform
    import argparse
    import pdb
    import pickle
    
    
    parser = argparse.ArgumentParser(description="preprocess_data: convert sames to python-friendly data format for faster loading")
    parser.add_argument("-a", "--already", help="data is already split into Test & Train (default is to add 80-20 split",action="store_true")
    parser.add_argument("-s", "--sequential", help="don't randomly shuffle data for train/test split",action="store_true")
    parser.add_argument("-m", "--mono", help="convert input audio to mono",action="store_true")
    parser.add_argument("-n", "--nosplit", help="do not create any Train/Test split in output (everything to same directory)",action="store_true")

    parser.add_argument("-r", "--resample", type=int, default=44100, help="convert input audio to mono")
    parser.add_argument('-d', "--dur",  type=float, default=3,   help='Max duration (in seconds) of each clip. Default = 3s')
    parser.add_argument('-c', "--clean", help="Assume 'clean data'; Do not check to find max shape (faster)", action='store_true')
    parser.add_argument('-f','--format', help="format of output file (npz, jpeg, png, etc). Default = npz", type=str, default='npz')
    parser.add_argument('-i','--inpath', help="input directory for audio samples (default='Samples')", type=str, default='Samples')
    parser.add_argument('--data', help="location of csv file containing tag information", type=str, default='../../data/data.csv')
    parser.add_argument('-o','--outpath', help="output directory for spectrograms (default='Preproc')", type=str, default='Preproc')
    parser.add_argument("--mels", help="number of mel coefficients to use in spectrograms", type=int, default=96)
    parser.add_argument("--phase", help="Include phase information as extra channels", action='store_true')

    args = parser.parse_args()
    if (('Darwin' == platform.system()) and (not args.mono)):
        # bug/feature in OS X that causes np.dot() to sometimes hang if multiprocessing is running
        mp.set_start_method('forkserver', force=True)   # hopefully this here makes it never hang
        print(" WARNING: Using stereo files w/ multiprocessing on OSX may cause the program to hang.")
        print(" This is because of a mismatch between the way Python multiprocessing works and some Apple libraries")
        print(" If it hangs, try running with mono only (-m) or the --clean option, or turn off parallelism")
        print("  See https://github.com/numpy/numpy/issues/5752 for more on this.")
        print("")

    yo = create_tag_dict(args.data)
    pickle.dump(yo, open('labels.p','wb'))

    preprocess_dataset(inpath=args.inpath+'/', outpath=args.outpath+'/', resample=args.resample, already_split=args.already, sequential=args.sequential, mono=args.mono,
        nosplit=args.nosplit, dur=args.dur, clean=args.clean, out_format=args.format, mels=args.mels, phase=args.phase)
