import MirCleaveNet
import sys, os
import numpy as np
import pandas as pd
import tensorflow as tf
from keras.models import load_model
from tensorflow.keras import Input
from tensorflow.keras.models import Model
from keras.layers import CuDNNLSTM, LSTM, Embedding, Dense, TimeDistributed, Dropout, Bidirectional
from MirCleaveNet_metrics import avg_proximity_metric
from seqeval.metrics import precision_score, recall_score, f1_score, classification_report
from scipy import stats
import math
import getopt


def print_usage():
    print("USAGE: %s <input file> [-m MODEL / -L ENSEMBLE_LIST] [-o OUTPUT_PREFIX] [-h]" %(sys.argv[0]));
    print("optional arguments:")
    print("-h, --help                  print usage");
    print("-o, --output_prefix         output prefix");
    print("-v, --verbose               verbose");
    print("-m, --model                 file containing weights for model");
    print("-L, --ensemble_list         file containing list of models");
    print("-d, --print_dvs             prints a list of decision values for label at each position");
    print("-i, --input_setting         set to one of the following");
    print("                            0 : USE_SEQ_ONLY")
    print("                            1 : USE_SEQ_FOLD")
    print("                            2 : USE_SEQ_BPRNA")
    

def load_input_output_file_parameters(parameters,opts={}):
    parameters["validation_file"] = opts['--validation_file'] if '--validation_file' in opts else None
    parameters["output_prefix"] = opts['--output_prefix'] if '--output_prefix' in opts else "results"
    parameters["predictions_file"] = parameters["output_prefix"] + "_predictions.txt"
    parameters["results_file"] = parameters["output_prefix"] + "_classificationReport.txt"
    parameters["param_out_file"] = parameters["output_prefix"] + "_parameters.txt"
    parameters["print_dvs"] = True if '--print_dvs' in opts else False
    if parameters["print_dvs"]:
        parameters["classification_DVs"] = parameters["output_prefix"] + "_classification_DVs.txt"
    return parameters

def print_metrics_file(perf,output_file):
    f = open(output_file,"w");
    metric_list = ["TP","FP","FN","TN","precision","recall","fscore","specificity","mcc","pmf","pse"]
    f.write("\t".join(["cutsite"] + metric_list) + "\n")
    for c in ["DR5","DC5","DC3","DR3"]:
        metric_values = [c] + [perf[c][m] for m in metric_list]
        f.write("{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\t{}\n".format(*metric_values))
    f.close()

def print_metrics_summary(perf):
    metric_list = ["precision","recall","fscore","pmf","pse"]
    print("\t".join(["cutsite"] + metric_list))
    for c in ["DR5","DC5","DC3","DR3"]:
        metric_values = [c] + [perf[c][m] for m in metric_list]
        print("{}\t{}\t{}\t{}\t{}\t{}".format(*metric_values))

def get_classification_metrics(y_vl,y_pred,parameters):
    idx_tag = {parameters['tag_idx'][k] : k for k in parameters['tag_idx']}
    perf = {}
    for c in parameters['tags']:
        if c != 'O':
            perf[c] = {d : 0 for d in ['TP','FP','FN','TN']}
    for i in range(0,y_pred.shape[0]):
        y_p = np.zeros_like(y_pred[i])
        y_t = y_vl[i]
        idx = y_pred[i].argmax(axis=0)
        for k in range(1,len(idx)):
            y_p[idx[k]][k] = 1
            tag = idx_tag[k]
            for j in range(0,y_p.shape[0]):
                if y_p[j][k] == y_t[j][k] and y_t[j][k] != 0:
                    perf[tag]['TP'] += 1
                elif y_p[j][k] != y_t[j][k] and y_t[j][k] == 0:
                    perf[tag]['FP'] += 1
                elif y_p[j][k] != y_t[j][k] and y_p[j][k] == 0:
                    perf[tag]['FN'] += 1
                elif y_p[j][k] == y_t[j][k] and y_t[j][k] == 0:
                    perf[tag]['TN'] += 1
                else:
                    print("error: get_performance\n",y_p[j][k],y_t[j][k]);
    for c in parameters['tags']:
        if c != 'O':
            if perf[c]['TP'] > 0:
                perf[c]['precision'] = perf[c]['TP'] / (perf[c]['TP'] + perf[c]['FP'])
                perf[c]['recall'] = perf[c]['TP'] / (perf[c]['TP'] + perf[c]['FN'])
                perf[c]['fscore'] = 2 * (perf[c]['precision'] * perf[c]['recall']) / (perf[c]['precision'] + perf[c]['recall'])
            else:
                perf[c]['precision'] = 0
                perf[c]['recall'] = 0
                perf[c]['fscore'] = 0
            perf[c]['specificity'] = perf[c]['TN'] / (perf[c]['TN'] + perf[c]['FP'])
            mcc_num = (perf[c]['TP'] * perf[c]['TN']) - (perf[c]['FP'] * perf[c]['FN'])
            mcc_denom_unsqr = (perf[c]['TP'] + perf[c]['FP']) 
            mcc_denom_unsqr *= (perf[c]['TP'] + perf[c]['FN'])
            mcc_denom_unsqr *= (perf[c]['TN'] + perf[c]['FP'])
            mcc_denom_unsqr *= (perf[c]['TN'] + perf[c]['FN'])
            perf[c]['mcc'] = mcc_num / math.sqrt(mcc_denom_unsqr)
    y_vl_argmax = np.array(y_vl).argmax(axis=1)
    y_pred_argmax = y_pred.argmax(axis=1)
    distances = np.absolute(y_pred_argmax - y_vl_argmax)
    for c in parameters['tags']:
        if c != 'O':
            idx = parameters['tag_idx'][c]
            perf[c]['pmf'] = ( distances.shape[0] - np.count_nonzero(distances[:,idx]) ) / float(distances.shape[0])
            perf[c]['pse'] = sum(distances[:,idx]) / float(distances.shape[0])
    return perf

def apply_weights(validation_pred,w):
    validation_pred_avg = np.zeros_like(validation_pred[0],dtype="float64")
    for i in range(len(validation_pred)):
        validation_pred_avg += validation_pred[i] * np.array(w[i])
    return validation_pred_avg

def read_predict_set(dataSetFile,parameters):
    dataSet = []
    f = open(dataSetFile,"r")
    for line in f.readlines():
        row_data = line.rstrip().split()
        seq_col = 2
        id,name,seq = row_data[0:seq_col+1]
        seq = MirCleaveNet.sanitizeSequence(seq)
        #adding dummy values for unneccessary fields
        mir_id = "-"
        prod5p = "-"
        prod3p = "-"
        drosha5p = MirCleaveNet.readCutSite("1,2")
        dicer5p = MirCleaveNet.readCutSite("1,2")
        dicer3p = MirCleaveNet.readCutSite("1,2")
        drosha3p = MirCleaveNet.readCutSite("1,2")
        hpStart = 1
        hpStop = 1
        if parameters["input_setting"] == "USE_SEQ_ONLY":
            dataSet.append([id,name,mir_id,prod5p,prod3p,drosha5p,dicer5p,dicer3p,drosha3p,int(hpStart),int(hpStop),seq])
        elif parameters["input_setting"] == "USE_SEQ_FOLD":
            fold = row_data[seq_col+1]
            dataSet.append([id,name,mir_id,prod5p,prod3p,drosha5p,dicer5p,dicer3p,drosha3p,int(hpStart),int(hpStop),seq,fold])            
        elif parameters["input_setting"] == "USE_SEQ_BPRNA":
            fold,bpRNA_context = row_data[seq_col+1:seq_col+3]
            fold = MirCleaveNet.add_context(fold,bpRNA_context)
            dataSet.append([id,name,mir_id,prod5p,prod3p,drosha5p,dicer5p,dicer3p,drosha3p,int(hpStart),int(hpStop),seq,fold])
        else:
            print("Error: INPUT_SETTING parameter is set to an invalid setting.  Valid settings are (\"USE_SEQ_ONLY\",\"USE_SEQ_FOLD\",\"USE_SEQ_BPRNA\").")
            exit()
    f.close()
    return dataSet

def print_predictions_output_file(dataSet,X_vl,predictions,predictions_outputFile,parameters):
    f = open(predictions_outputFile,"w")
    DR5_idx = parameters['tag_idx']['DR5']
    DC5_idx = parameters['tag_idx']['DC5']
    DC3_idx = parameters['tag_idx']['DC3']
    DR3_idx = parameters['tag_idx']['DR3']
    predictions_argmax = np.array(predictions).argmax(axis=1)    
    #pred_labels = MirCleaveNet.pred2label(predictions,parameters)
    f.write("#id\tname\tDR5\tDC5\tDC3\tDR3\n")
    for i in range(0,len(dataSet)):
        id,name,seq = dataSet[i][0:3]
        #cutsites are 1-based
        DR5_argmax = predictions_argmax[i][DR5_idx] + 1
        DC5_argmax = predictions_argmax[i][DC5_idx] + 1
        DC3_argmax = predictions_argmax[i][DC3_idx] + 1
        DR3_argmax = predictions_argmax[i][DR3_idx] + 1
        DR5_cutsite = "%d,%d" % (DR5_argmax,DR5_argmax+1)
        DC5_cutsite = "%d,%d" % (DC5_argmax,DC5_argmax+1)
        DR3_cutsite = "%d,%d" % (DR3_argmax,DR3_argmax+1)
        DC3_cutsite = "%d,%d" % (DC3_argmax,DC3_argmax+1)
        f.write("%s\n"%("\t".join([id,name,DR5_cutsite,DC5_cutsite,DC3_cutsite,DR3_cutsite])))
    f.close();


def poly_loss(y_true, y_pred, alpha=0.25, gamma=1.0):
    """
    PolyLoss is a modification of the focal loss that can be used to address class imbalance in classification tasks.
    
    Parameters:
    - y_true: Ground truth labels (one-hot encoded).
    - y_pred: Predicted probabilities.
    - alpha: Weighting factor for the modulating term (default: 0.25).
    - gamma: Focusing parameter to down-weight easy examples (default: 2.0).
    
    Returns:
    - loss: Computed PolyLoss value.
    """
    # Clip predictions to avoid log(0)
    y_pred = K.clip(y_pred, K.epsilon(), 1. - K.epsilon())

    # Compute cross-entropy
    cross_entropy = -y_true * K.log(y_pred)

    # Compute the modulating factor (1 - p_t)^gamma
    modulating_factor = K.pow(1 - y_pred, gamma)

    # Apply PolyLoss formulation
    poly_loss = alpha * modulating_factor * cross_entropy
    return K.sum(poly_loss, axis=-1)
if __name__ == "__main__":
    
    min_args = 2
    if (len(sys.argv) < min_args):
        print_usage()
        exit()
    try:
        input_file = sys.argv[1]
        opts_array, args = getopt.getopt(sys.argv[min_args:], "hvdo:e:m:i:L:", ["help","verbose","print_dvs","output_prefix=","epochs=","model=","input_setting=","max_seq_len=",
                                                                               "use_embedding_layer=","use_crf_layer=","embedding_layer_output=","embedding_dropout=","bi_lstm1_units=","bi_lstm2_units=",
                                                                                "dense1_units=","dense2_units=","ensemble_list="])
        if len(args) > 0:
            print("%s is not an option\n"%(args[0]))
            print_usage()
            exit()
        longopts_map = {
            "-h" : "--help",
            "-d" : "--print_dvs",
            "-o" : "--output_prefix",
            "-e" : "--epochs",
            "-v" : "--verbose",
            "-m" : "--model",
            "-i" : "--input_setting",
            "-L" : "--ensemble_list",
        }
        opts = {longopts_map[o] : a for o,a in opts_array if o in longopts_map}
        opts.update({o : a for o,a in opts_array if o not in longopts_map})
        opts["--validation_file"] = input_file
    except getopt.GetoptError as err:
        print(err)
        print_usage()
        exit()
    if '--help' in opts:
        print_usage()
        exit()
    
    parameters = MirCleaveNet.load_parameters(opts)
    parameters = load_input_output_file_parameters(parameters,opts)

    predictions_outputFile = parameters["output_prefix"] + "_predicted_cutsites.txt"

    inputSet = read_predict_set(input_file,parameters)
    new_inputSet = MirCleaveNet.dropLongSequences(inputSet,parameters)
    X_vl,_ = MirCleaveNet.prepareData(new_inputSet,parameters)
    if "ensemble" in parameters:
        print(1)
        predictions = []
        for i in range(0,len(parameters["ensemble"])):
            print("loading model %d: %s\n"%(i, parameters["ensemble"][i]))
            model = load_model(parameters["ensemble"][i], custom_objects={'prox':avg_proximity_metric()})
            predictions.append(model.predict(X_vl, verbose=parameters["verbose"]))
        predictions_avg = apply_weights(predictions,w=parameters["ensemble_weights"])
        print_predictions_output_file(new_inputSet,X_vl,predictions_avg,predictions_outputFile,parameters)
        if parameters["print_dvs"]:
            MirCleaveNet.print_classification_values_file(new_inputSet,predictions_avg,parameters,output_file=parameters["classification_DVs"])
    else:
        print(2)
        model = load_model(parameters["model"], custom_objects={'prox':avg_proximity_metric(),'poly_loss':poly_loss})
        model.summary()
        predictions = model.predict(X_vl, verbose=parameters["verbose"])    
        print_predictions_output_file(new_inputSet,X_vl,predictions,predictions_outputFile,parameters)
        if parameters["print_dvs"]:
            MirCleaveNet.print_classification_values_file(new_inputSet,predictions,parameters,output_file=parameters["classification_DVs"])
