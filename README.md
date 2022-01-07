# GreThE
A Dataset for Speech Emotion Recognition in Greek Theatrical Plays

## 1 Dataset format
### 1.1 Tasks

The Arousal and Valence tasks are provided in a classification format under three classes: _(i)_ weak, _(ii)_ neutral, _(iii)_ strong for arousal and _(i)_ negative, _(ii)_ neutral, _(iii)_ positive for valence.   

### 1.2 Available features
Three types of npy files are provided for each audio file:
1. Mel-spectrograms
2. a sequence of 68 segment features calculated by [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)
3. a sequence of segment statistics (136 features) calculated by [pyAudioAnalysis](https://github.com/tyiannak/pyAudioAnalysis)

## 2 Filenames
Every feature filename is in the form of \<id1>_speaker\<id2>-\<id3>.npy where:
- id1 is the session (ie. theatrical play) id
- id2 is the speaker id
- id3 is the utterance id for a specific speaker and session
  
## 3 Feature extraction

In order to perform the same feature extraction procedure on new raw audio files, you can use the ```get_melgram``` and ```pyaudio_segment_features``` functions found in feature_extraction.py

Note that the raw audio files must be mono and have a sampling rate of 8K. 

## 4 Cite

To be filled
