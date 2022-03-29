from deep_audio_features.bin import basic_test
import argparse
import numpy 
import pickle
from deep_audio_features.models.cnn import load_cnn


def read_class_names(model_name):
    with open(model_name, "rb") as input_file:
        model_params = pickle.load(input_file)
    if "classes_mapping" in model_params:
        task = "classification"
        model, hop_length, window_length = load_cnn(model_name)
        class_names = model.classes_mapping
    return class_names


def get_probs_from_posteriors(posteriors):
    posts = numpy.array(posteriors)
    probs = []
    for w in range(posts.shape[0]): # for each segment:
        p = numpy.exp(posts[w, :]) / numpy.sum(numpy.exp(posts[w, :]))
        probs.append(p)
    probs = numpy.array(probs)
    return probs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i',
                        '--input',
                        nargs='+',
                        required=True,
                        help='Files to analyze')
    FLAGS = parser.parse_args()

    input = FLAGS.input
    arousal = "merged_3sec_arousal_transfered_fin.pt"
    valence = "merged_3sec_valence_transfered_fin.pt"

    arousal_preds = []
    valence_preds = []

    class_names_ar = read_class_names(arousal)
    class_names_va = read_class_names(valence)
    preds, posteriors_ar = basic_test.test_model(arousal, input[0], layers_dropped=0,
               test_segmentation=True, verbose=True)    
    probs_ar = get_probs_from_posteriors(posteriors_ar)

    preds, posteriors_va = basic_test.test_model(valence, input[0], layers_dropped=0,
               test_segmentation=True, verbose=True)    
    probs_va = get_probs_from_posteriors(posteriors_va)

    print(probs_ar)
    print(class_names_ar)
    classes_vals_ar = list(class_names_ar.keys())
    classes_names_ar = list(class_names_ar.values())

    print(probs_va)
    print(class_names_va)
    classes_vals_va = list(class_names_va.keys())
    classes_names_va = list(class_names_va.values())

    for i in range(len(probs_ar)):
        arousal_preds.append(probs_ar[i][classes_vals_ar[classes_names_ar.index("strong")]] - probs_ar[i][classes_vals_ar[classes_names_ar.index("weak")]])

    for i in range(len(probs_va)):
        valence_preds.append(probs_va[i][classes_vals_va[classes_names_va.index("positive")]] - probs_va[i][classes_vals_va[classes_names_va.index("negative")]])

    import matplotlib.pyplot as plt
    plt.plot(valence_preds, arousal_preds, '*r')
    plt.show()