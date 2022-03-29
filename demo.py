from deep_audio_features.bin import basic_test
import argparse
import numpy as np
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
    posts = np.array(posteriors)
    probs = []
    for w in range(posts.shape[0]): # for each segment:
        p = np.exp(posts[w, :]) / np.sum(np.exp(posts[w, :]))
        probs.append(p)
    probs = np.array(probs)
    return probs


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input', nargs='+',
                        required=True, help='Files to analyze')
    FLAGS = parser.parse_args()

    input = FLAGS.input
    arousal = "models/merged_3sec_arousal_transfered_fin.pt"
    valence = "models/merged_3sec_valence_transfered_fin.pt"
    general = "models/4_class.pt"

    arousal_preds = []
    valence_preds = []

    class_names_ar = read_class_names(arousal)
    class_names_va = read_class_names(valence)
    class_names_4 = read_class_names(general)

    preds, posteriors_ar = basic_test.test_model(arousal, input[0], layers_dropped=0,
               test_segmentation=True, verbose=True)    
    probs_ar = get_probs_from_posteriors(posteriors_ar)

    preds, posteriors_va = basic_test.test_model(valence, input[0], layers_dropped=0,
               test_segmentation=True, verbose=True)    
    probs_va = get_probs_from_posteriors(posteriors_va)

    preds, posteriors_4 = basic_test.test_model(general, input[0], layers_dropped=0,
               test_segmentation=True, verbose=True)    
    probs_4 = get_probs_from_posteriors(posteriors_4)


    classes_vals_ar = list(class_names_ar.keys())
    classes_names_ar = list(class_names_ar.values())
    classes_vals_va = list(class_names_va.keys())
    classes_names_va = list(class_names_va.values())
    classes_vals_4 = list(class_names_4.keys())
    classes_names_4 = list(class_names_4.values())

    print(probs_ar)
    print(class_names_ar)
    print(probs_va)
    print(class_names_va)
    print(probs_4)
    print(class_names_4)

    speech_probs = []
    for i in range(len(probs_4)):
        p = classes_names_4.index("speech")
        speech_probs.append(probs_4[i][p])

    print(speech_probs)

    for i in range(len(probs_ar)):
        cur_speech_prob = np.mean(speech_probs[i * 3: i * 3 + 3])
        if cur_speech_prob > 0.6:
            arousal_preds.append(probs_ar[i][classes_vals_ar[classes_names_ar.index("strong")]] - probs_ar[i][classes_vals_ar[classes_names_ar.index("weak")]])

    for i in range(len(probs_va)):
        cur_speech_prob = np.mean(speech_probs[i * 3: i * 3 + 3])
        if cur_speech_prob > 0.6:
            valence_preds.append(probs_va[i][classes_vals_va[classes_names_va.index("positive")]] - probs_va[i][classes_vals_va[classes_names_va.index("negative")]])
#    import pdb; pdb.set_trace()
    import pandas as pd
    valence_preds.append(-0.95)
    valence_preds.append(-.95)
    valence_preds.append(.95)
    valence_preds.append(.95)
    arousal_preds.append(-.95)
    arousal_preds.append(.95)
    arousal_preds.append(-.95)
    arousal_preds.append(.95)

    v = pd.Series(valence_preds)
    aa = pd.Series(arousal_preds)
    df = pd.DataFrame({'valence': v, 'arousal': aa})
    
    import plotly.express as px
    fig = px.density_heatmap(df, x="valence", y="arousal", nbinsx=10, nbinsy=10)

    ff=dict(family="Courier New, monospace",
            size=22,
            color="#eeffff")

    fig.add_annotation(text="happiness", x=0.3, y=0.5, arrowhead=1, showarrow=True, font=ff, arrowwidth=2, arrowcolor="#eeffff")    
    fig.add_annotation(text="calmness", x=0.3, y=-0.5, arrowhead=1, showarrow=True, font=ff, arrowwidth=2, arrowcolor="#eeffff")    
    fig.add_annotation(text="sadness", x=-0.3, y=-0.5, arrowhead=1, showarrow=True, font=ff, arrowwidth=2, arrowcolor="#eeffff")    
    fig.add_annotation(text="fear", x=-0.5, y=0.2, arrowhead=1, showarrow=True, font=ff, arrowwidth=2, arrowcolor="#eeffff")    
    fig.add_annotation(text="anger", x=-0.3, y=0.5, arrowhead=1, showarrow=True, font=ff, arrowwidth=2, arrowcolor="#eeffff")
    fig.add_annotation(text="excitement", x=0.1, y=0.6, arrowhead=1, showarrow=True, font=ff, arrowwidth=2, arrowcolor="#eeffff")
    fig.show()
#    import matplotlib.pyplot as plt
#    plt.plot(valence_preds, arousal_preds, '*r')
#    plt.show()