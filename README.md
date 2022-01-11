# GreThE
A Dataset for Speech Emotion Recognition in Greek Theatrical Plays

## 1. General Info
| Number of samples  | 500  |
| Total duration  | 46 mins  |
| Classification tasks annotated  | 2 (valence and arousal)  |
| Number of human annotators  | 4  |
| Number of theatrical plays used | 23  |
| Number of unique speakers | 90  |
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
The `evaluation.py` srcipt is provided in order to perform a basic session-independent evaluation using feature statistics and [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis). <br/>
For arousal, run: <br/>
```python3 evaluation.py -p data/arousal/pyaudio/segment_stats/weak data/arousal/pyaudio/segment_stats/neutral data/arousal/pyaudio/segment_stats/strong```
<br/>
For valence, run: <br/>
```python3 evaluation.py -p data/valence/pyaudio/segment_stats/negative data/valence/pyaudio/segment_stats/neutral data/valence/pyaudio/segment_stats/positive```

## 5. Cite
To be filled
