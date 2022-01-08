from pyAudioAnalysis import audioTrainTest as att
import os
import argparse
import glob
import numpy as np
from sklearn.preprocessing import StandardScaler    
import pickle as cPickle


def make_group_list(filenames):
    '''
    This function is responsible for creating group id of every sample according to the theatrical show
    :param filenames: list of samples' names
    :return: list of group ids
    '''
    groups_id = []
    groups_name = []
    id = 1
    for f in filenames:
        show_id = f.split('/')[-1].split('_')[0]
        print(show_id)
        found = False
        for count, gr in enumerate(groups_id):
            if show_id == groups_name[count]:
                groups_id.append(gr)
                groups_name.append(show_id)
                found = True
                break
        if found == False:
            groups_id.append(id)
            groups_name.append(show_id)
            id += 1
    return groups_id


def load_features(path_list):
    features = []
    class_names = []
    file_names = []
    features = []
    for i, d in enumerate(path_list):    
        mid_term_features = []
        for file in os.listdir(d):
            if file.endswith(".npy"):
                filename = os.path.join(d, file)
                file_names.append(filename)

                mid_features = np.load(filename)
                mid_features = np.transpose(mid_features)
                mid_features = mid_features.mean(axis=0)
                
                # long term averaging of mid-term statistics
                if (not np.isnan(mid_features).any()) and \
                        (not np.isinf(mid_features).any()):
                    if len(mid_term_features) == 0:
                        # append feature vector
                        mid_term_features = mid_features
                    else:
                        mid_term_features = np.vstack((mid_term_features, mid_features))
        
        features.append(mid_term_features)   
        if d[-1] == os.sep:
            class_names.append(d.split(os.sep)[-2])
        else:
            class_names.append(d.split(os.sep)[-1])

    return features, class_names, file_names


def extract_features_and_train(
    paths, classifier_type, model_name,
    compute_beat=False, train_percentage=0.90,
    dict_of_ids=None):
    """
    This function is used as a wrapper to segment-based audio feature extraction
    and classifier training.
    ARGUMENTS:
        paths:                      list of paths of directories. Each directory
                                    contains a signle audio class whose samples
                                    are stored in seperate WAV files.
        classifier_type:            "svm" or "knn" or "randomforest" or
                                    "gradientboosting" or "extratrees"
        model_name:                 name of the model to be saved
        dict_of_ids:                a dictionary which has as keys the full path of audio files and as values the respective group ids
    RETURNS:
        None. Resulting classifier along with the respective model
        parameters are saved on files.
    """
    # STEP A: Feature Extraction:
    features, class_names, file_names = \
        load_features(paths)
    #file_names = [item for sublist in file_names for item in sublist]
    if dict_of_ids:
        list_of_ids = [dict_of_ids[file] for file in file_names]
    else:
        list_of_ids = None
    if len(features) == 0:
        print("trainSVM_feature ERROR: No data found in any input folder!")
        return

    n_feats = features[0].shape[1]
    feature_names = ["features" + str(d + 1) for d in range(n_feats)]

    for i, feat in enumerate(features):
        if len(feat) == 0:
            print("trainSVM_feature ERROR: " + paths[i] +
                  " folder is empty or non-existing!")
            return

    # STEP B: classifier Evaluation and Parameter Selection:
    if classifier_type == "svm" or classifier_type == "svm_rbf":
        classifier_par = np.array([0.001, 0.01,  0.5, 1.0, 5.0, 10.0, 20.0])
    elif classifier_type == "randomforest":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    elif classifier_type == "knn":
        classifier_par = np.array([1, 3, 5, 7, 9, 11, 13, 15])        
    elif classifier_type == "gradientboosting":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])
    elif classifier_type == "extratrees":
        classifier_par = np.array([10, 25, 50, 100, 200, 500])

    # get optimal classifier parameter:
    temp_features = []
    for feat in features:
        temp = []
        for i in range(feat.shape[0]):
            temp_fv = feat[i, :]
            if (not np.isnan(temp_fv).any()) and (not np.isinf(temp_fv).any()):
                temp.append(temp_fv.tolist())
            else:
                print("NaN Found! Feature vector not used for training")
        temp_features.append(np.array(temp))
    features = temp_features

    best_param = att.evaluate_classifier(
        features, class_names, classifier_type,
        classifier_par, parameter_mode=0,
        list_of_ids=list_of_ids, n_exp=-1,
        train_percentage=train_percentage)

    print("Selected params: {0:.5f}".format(best_param))


    # STEP C: Train and Save the classifier to file

    # First Use mean/std standard feature scaling:
    features, labels = att.features_to_matrix(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    mean = scaler.mean_.tolist()
    std = scaler.scale_.tolist()

    # Then train the final classifier
    if classifier_type == "svm":
        classifier = att.train_svm(features, labels, best_param)
    elif classifier_type == "svm_rbf":
        classifier = att.train_svm(features, labels, best_param, kernel='rbf')
    elif classifier_type == "randomforest":
        classifier = att.train_random_forest(features, labels, best_param)
    elif classifier_type == "gradientboosting":
        classifier = att.train_gradient_boosting(features, labels, best_param)
    elif classifier_type == "extratrees":
        classifier = att.train_extra_trees(features, labels, best_param)
    
    return classifier, mean, std

def train_val(paths, classifier_type, model_name):
    listOfFile = []
    types = ('*.npy')

    for path in paths:
        for files in types:
            listOfFile.extend(glob.glob(os.path.join(path, files)))
    groups = make_group_list(listOfFile)
    group_dict = dict(zip(listOfFile, groups))
    extract_features_and_train(paths, classifier_type, model_name, dict_of_ids=group_dict)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-p',
                        '--paths',
                        type=str,
                        nargs='+',
                        required=True,
                        help='Input class folders')
    FLAGS = parser.parse_args()
    classifier_type = 'svm_rbf'
    model_name = 'theatrical_emotion'
    train_val(FLAGS.paths, classifier_type, model_name)