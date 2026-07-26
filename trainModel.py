import MirCleaveNet
import sys, os
import numpy as np
import pandas as pd
from keras_contrib.utils import save_load_utils
from keras.callbacks import EarlyStopping
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
import getopt
import tensorflow as tf
import pickle
from collections import Counter
from tensorflow.keras.callbacks import Callback
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras.callbacks import ModelCheckpoint
def print_usage():
    print("USAGE: %s <train file> [-e EPOCHS] [-o OUTPUT_PREFIX] [-h]" %(sys.argv[0]));
    print("optional arguments:")
    print("-h, --help                  print usage");
    print("-e, --epochs                number of epochs");
    print("-p, --patience              patience during training")
    print("-s, --validation_file       validation set file")
    print("-o, --output_prefix         output prefix");
    print("-v, --verbose               verbose");
    print("-m, --model                 file containing weights for model");
    print("-d, --print_dvs             prints a list of decision values for label at each position");
    print("-i, --input_setting         set to one of the following");
    print("                            0 : USE_SEQ_ONLY")
    print("                            1 : USE_SEQ_FOLD")
    print("                            2 : USE_SEQ_BPRNA")
    print("")
    #print("--batch_size                Batch Size")
    #print("--max_seq_len               Maximum sequence length")
    #print("--use_embedding_layer       Train with an embedding layer (True/False)")
    #print("--use_crf_layer             Train with a crf layer (True/False)")
    print("--embedding_layer_output    Embedding layer output size")
    print("--embedding_dropout         Dropout for Embedding layer")
    print("--bi_lstm1_units            units for first bidirectional lstm")
    print("--bi_lstm2_units            units for second bidirectional lstm")
    #print("--dense1_units              units for first dense layer")
    #print("--dense2_units              units for second dense layer")
    print("--learning rate             learning_rate for adam optimizer")
    print("--epsilon                   epsilon for adam optimizer")
    #print("--pickle                    create pickle file of results for validation set")

def load_input_output_file_parameters(parameters,opts={}):
    parameters["train_file"] = opts['--train_file']
    parameters["validation_file"] = opts['--validation_file'] if '--validation_file' in opts else None
    parameters["output_prefix"] = opts['--output_prefix'] if '--output_prefix' in opts else "results"
    parameters["predictions_file"] = parameters["output_prefix"] + "_validation_predictions.txt"
    parameters["results_file"] = parameters["output_prefix"] + "_classificationReport.txt"
    parameters["acc_fig"] = parameters["output_prefix"] + "_acc.svg"
    parameters["loss_fig"] = parameters["output_prefix"] + "_loss.svg"
    parameters["prox_fig"] = parameters["output_prefix"] + "_prox.svg"
    parameters["param_out_file"] = parameters["output_prefix"] + "_parameters.txt"
    parameters["history_file"] = parameters["output_prefix"] + ".history"
    parameters["print_dvs"] = True if '--print_dvs' in opts else False
    if parameters["print_dvs"]:
        parameters["classification_DVs"] = parameters["output_prefix"] + "_classification_DVs.txt"
    parameters["pickle"] = True if '--pickle' in opts else False
    if parameters["pickle"]:
        parameters["pickle_file"] = parameters["output_prefix"] + "_validation_results.pickle"

    return parameters

def print_history(history,output_file):
    f = open(output_file,"w")
    f.write("epoch\tloss\tacc\tprox\tval_loss\tval_acc\tval_prox\n")
    for i in range(len(history.history['loss'])):
        prox = history.history['prox'][i] if 'prox' in history.history else 'na'
        val_prox = history.history['val_prox'][i] if 'val_prox' in history.history else 'na'
        f.write("%d\t%f\t%f\t%s\t%f\t%f\t%s\n"%(i+1,history.history['loss'][i],history.history['accuracy'][i],str(prox),history.history['val_loss'][i],history.history['val_accuracy'][i],str(val_prox)))
    f.close()

def print_parameters(parameters,param_out_file):
    f = open(param_out_file,"w")
    for key in parameters:
        f.write("%s=%s\n"%(key,str(parameters[key])))
    f.close() 




if __name__ == "__main__":
    
    min_args = 2
    if (len(sys.argv) < min_args):
        print_usage()
        exit()
    try:
        train_file = sys.argv[1]
        opts_array, args = getopt.getopt(sys.argv[min_args:], "hvdo:e:s:m:i:p:", ["help","verbose","output_prefix=","epochs=","validation_file=","model=","input_setting=","max_seq_len=","patience=",
                                                                               "use_embedding_layer=","use_crf_layer=","embedding_layer_output=","embedding_dropout=","bi_lstm1_units=","bi_lstm2_units=",
                                                                                  "dense1_units=","dense2_units=","batch_size=","print_dvs","pickle","learning_rate=","epsilon="])
        if len(args) > 0:
            print("%s is not an option\n"%(args[0]))
            print_usage()
            exit()
        longopts_map = {
            "-h" : "--help",
            "-d" : "--print_dvs",
            "-o" : "--output_prefix",
            "-e" : "--epochs",
            "-p" : "--patience",
            "-v" : "--verbose",
            "-s" : "--validation_file",
            "-m" : "--model",
            "-i" : "--input_setting",
        }
        opts = {longopts_map[o] : a for o,a in opts_array if o in longopts_map}
        opts.update({o : a for o,a in opts_array if o not in longopts_map})
        opts["--train_file"] = train_file
    except getopt.GetoptError as err:
        print(err)
        print_usage()
        exit()
    if '--help' in opts:
        print_usage()
        exit()
    
    parameters = MirCleaveNet.load_parameters(opts)
    parameters = load_input_output_file_parameters(parameters,opts)

    print("patience=",parameters["patience"])
    trainSet = MirCleaveNet.readDataset(parameters["train_file"],parameters) 
    new_trainSet = MirCleaveNet.dropLongSequences(trainSet,parameters) 
    X_tr,y_tr = MirCleaveNet.prepareData(new_trainSet,parameters) 
    
    y_tr_labels = np.argmax(np.array(y_tr), axis=-1).flatten()

    print(Counter(y_tr_labels))
    model = MirCleaveNet.createArchitecture(parameters)

    
    if parameters["validation_file"]:
        validationSet = MirCleaveNet.readDataset(parameters["validation_file"],parameters)
        new_validationSet = MirCleaveNet.dropLongSequences(validationSet,parameters)
        X_vl,y_vl = MirCleaveNet.prepareData(new_validationSet,parameters)
        if len(np.array(y_vl).shape) == 3:
            y_vl_labels = np.argmax(np.array(y_vl), axis=-1).flatten()
        else:
            y_vl_labels = np.array(y_vl).flatten()

        print(Counter(y_vl_labels))
        validation_labels = MirCleaveNet.pred2label(y_vl,parameters)
        # es = EarlyStopping(monitor='val_loss', min_delta=0, patience=parameters["patience"], verbose=parameters["verbose"], mode='min', restore_best_weights=True)
        es = EarlyStopping(monitor='val_prox', min_delta=0, patience=parameters["patience"], verbose=parameters["verbose"], mode='min', restore_best_weights=True)
        #mc = ModelCheckpoint(filepath="model_epoch7/model_epoch_{epoch:02d}",monitor="val_prox",mode="min",save_best_only=False,save_weights_only=False,verbose=parameters["verbose"])
        history = model.fit(X_tr, np.array(y_tr),batch_size=parameters["batch_size"],epochs=parameters["epochs"],
                            validation_data=(X_vl,np.array(y_vl)),verbose=1,shuffle=True,callbacks=[es,mc])  
        print_history(history,parameters['history_file'])
        validation_pred = model.predict(X_vl, verbose=parameters["verbose"])
        pred_labels = MirCleaveNet.pred2label(validation_pred,parameters)
        MirCleaveNet.print_validation_output_file(new_validationSet,X_vl,validation_labels,pred_labels,parameters["predictions_file"],parameters)
        MirCleaveNet.create_history_fig(history,parameters["acc_fig"],parameters["loss_fig"],parameters["prox_fig"],parameters)
        MirCleaveNet.print_classification_report(validation_labels,pred_labels,outputFile=parameters["results_file"])
        #print("F1-score: {:.1%}".format(f1_score(validation_labels, pred_labels)))
        #print(classification_report(validation_labels, pred_labels))
        if parameters["print_dvs"]:
            MirCleaveNet.print_classification_values_file(new_validationSet,validation_pred,parameters,output_file=parameters["classification_DVs"])
        if parameters["pickle"]:
            results = {
                "validationSet" : new_validationSet,
                "validation_DV" : validation_pred,
                "validation_yTrue" : validation_labels,
                "validation_yPred" : pred_labels,
            }
            f = open(parameters["pickle_file"], 'wb')
            pickle.dump(results,f)
            f.close()
        print_parameters(parameters,parameters["param_out_file"])
    else:
        history = model.fit(X_tr, np.array(y_tr),batch_size=parameters["batch_size"],epochs=parameters["epochs"],verbose=parameters["verbose"],shuffle=True)        

    model.save(parameters["model"])
