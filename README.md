# GreThE
> Developed at [**MagCIL**](https://magcil.github.io), the Multimodal Analysis Group of the Institute of Informatics and Telecommunications, [NCSR "Demokritos"](https://www.demokritos.gr/), Athens, Greece.

A Dataset for Speech Emotion Recognition in Greek Theatrical Plays

## 1. General Info
| Info  |   |
| --- | --- |
| # samples  | 500  |
| Total duration  | 46 mins  |
| # Classification tasks  | 2 (valence and arousal)  |
| # human annotators  | 4  |
| # theatrical plays used | 23  |
| # unique speakers | 90  |
| Language | Greek  |

## 2 Dataset format
### 2.1 Classification Tasks
The Arousal and Valence tasks are provided in a classification format under three classes: _(i)_ weak, _(ii)_ neutral, _(iii)_ strong for arousal and _(i)_ negative, _(ii)_ neutral, _(iii)_ positive for valence.   

### 2.2 Available features
Three types of numpy binaries (`npy` files) are provided for each audio sample:
1. Mel-spectrograms
2. a sequence of 68 segment feature vectors calculated by [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis), using a 50 msec non overlapping window. In other words, each utterance is represented by a number_of_frames x 68 short-term features. 
3. a sequence of segment statistics (i.e. the mean and std of the 68 short-term fetures, that is a 136-D feature vector for the whole utterance) calculated by [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis). 

### 2.3 Filenames and metadata
The filenames of the examples are in the form of \<id1>_speaker\<id2>-\<id3>.npy where:
- id1 is the session (ie. theatrical play) id
- id2 is the speaker id
- id3 is the utterance id for a specific speaker and session

This information can be used for session depedent cross-validation.

## 3. Feature extraction
In order to perform the same feature extraction procedure on new (unseen) raw audio files, you can use the ```get_melgram``` and ```pyaudio_segment_features``` functions found in `feature_extraction.py`

Note that the raw audio files must be mono and have a sampling rate of 8K. 

## 4. Basic Evaluation
The `evaluation.py` srcipt is provided in order to perform a basic session-independent evaluation using feature statistics and [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis).

First install dependencies by: 
```
pip3 install -r requirements.txt
```

For arousal, run: 
```
python3 evaluation.py -p data/arousal/pyaudio/segment_stats/weak data/arousal/pyaudio/segment_stats/neutral data/arousal/pyaudio/segment_stats/strong
```

For valence, run: 
```
python3 evaluation.py -p data/valence/pyaudio/segment_stats/negative data/valence/pyaudio/segment_stats/neutral data/valence/pyaudio/segment_stats/positive
```

## 5. Cite
To be filled

## About MagCIL

This repository comes from [MagCIL](https://magcil.github.io), the Multimodal Analysis Group of the
Institute of Informatics and Telecommunications at the National Centre for Scientific Research
"Demokritos" in Athens, Greece. We work on multimodal deep learning for audio, speech, music and
video, from fundamental research to deployed systems.

- Group website: <https://magcil.github.io>
- Publications: <https://magcil.github.io/pub.html>
- All our open-source code and data: <https://magcil.github.io/code.html>
- Work with us, in industry projects or research consortia: <https://magcil.github.io/collaborate.html>

Other tools from the group:

- [`pyAudioAnalysis`](https://github.com/tyiannak/pyAudioAnalysis) &mdash; audio feature extraction, classification, segmentation and applications
- [`deep_audio_features`](https://github.com/tyiannak/deep_audio_features) &mdash; training and using CNNs for audio classification (PyTorch)
- [`deepaudio-x`](https://github.com/magcil/deepaudio-x) &mdash; a PyTorch framework for audio classification
- [`deepaudio-lab`](https://github.com/magcil/deepaudio-lab) &mdash; a no-code platform for training, evaluating and deploying audio classifiers
- [`amvoc`](https://github.com/tyiannak/amvoc) &mdash; analysis of mouse vocal communication
- [`paura`](https://github.com/tyiannak/paura) &mdash; real-time audio recording and analysis

The full list is at <https://github.com/magcil>.
