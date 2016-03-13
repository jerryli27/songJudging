
# coding: utf-8

# In[27]:

import IPython, numpy as np, scipy as sp, matplotlib.pyplot as plt, matplotlib, sklearn, librosa, cmath,math
from IPython.display import Audio
 


# In[28]:

def estimated_beat(wav_file, hop_size, start_time, end_time, sr, start_bpm):
    signal, sr = librosa.load(wav_file, sr)
    hop_length = 512
    start = start_time
    end = end_time
    tempo, beats = librosa.beat.beat_track(y=signal, sr=sr, onset_envelope=None, hop_length=hop_size,
                start_bpm=start_bpm, tightness=100, trim=True, bpm=None)
    onset_env = librosa.onset.onset_strength(y=signal, sr=sr, S=None, lag=1, max_size=1, detrend=False, 
                                 center=True, feature=None, aggregate=None)
    return onset_env, beats
# using correlation or fftconvolution to get alignment
# However, this takes way too long, so we will assume starting and end time are the same. 
# def calculate_rank(original_signal, test_signal, original_beats, test_beats, sr, sigma):
#     original_signal, sr = librosa.load(original_signal, sr)
#     test_signal, sr = librosa.load(test_signal, sr)
#     original = np.zeros(len(original_signal))
#     test = np.zeros(len(test_signal))
#     for i in range (0, len(original_beats)):
#         original[original_beats[i]] = 1.0
#     for i in range (0, len(test_beats)):
#         test[test_beats[i]] = 1.0
#     reversed_test = test[::-1]
#     correlation = np.correlate(original, reversed_test, "full")
#     index = np.argmax(correlation) 
#     print correlation
#     if (index < len(test)):
#         correlated_test = np.zeros(len(test))
#         for i in range (0, index + 1):
#             correlated_test[i] = test[(len(test) - index - 1) + i]
#     else:
#         correlated_test = np.zeros(len(test)+(index-len(test)))
#         for i in range (0, len(test)):
#             correlated_test[i+(index-len(test))] = test[i]

def calculate_rank(original_beats, test_beats, sr, offbeat_factor):
    score_array = np.zeros(len(original_beats))
    for i in range (0, len(original_beats)):
        for k in range (0, len(test_beats)):
            if ((original_beats[i] + offbeat_factor > test_beats[k]) and (original_beats[i] - offbeat_factor < test_beats[k])):
                score_array[i] = 1.0
    score = np.sum(score_array)/float(len(score_array))
    score = score*100.0
    return score 


# In[29]:

def analyze_signals(original_signal, test_signal, hop_size, start_time, end_time, sr, start_bpm, offbeat_factor):
    # Offbeat_factor is how much off the beat the person is allowed to be
    test, sr = librosa.load(test_signal, sr)
    test_normalizer = np.max(test)
    test = test/float(test_normalizer)
    original, sr = librosa.load(original_signal, sr)
    original_normalizer = np.max(original)
    original = original/float(original_normalizer)
    test_onset_env, test_beat_frames = estimated_beat(test_signal, hop_size, start_time, end_time, sr, start_bpm)
    original_onset_env, original_beat_frames = estimated_beat(original_signal, hop_size, start_time, end_time, sr, start_bpm)
    # plt.show()
    test_beats = librosa.frames_to_samples(test_beat_frames, hop_length=hop_size)
    original_beats = librosa.frames_to_samples(original_beat_frames, hop_length=hop_size)
    beat_score = calculate_rank(original_beats, test_beats, sr, sr*offbeat_factor)
    
    plt.figure(1)
    ax1 = plt.subplot(2,1,1)
    plt.plot(original, label='Signal')
    plt.vlines(original_beats, -2, 2, alpha=.5, color='r',
                linestyle='solid', linewidth=3, label='Beats')
    plt.legend(frameon=True, framealpha=0.75)
    # Limit the plot to a X-second window
    plt.xlim([start_time * sr, end_time * sr])
    plt.xticks(np.linspace(start_time, end_time, 5) * sr,
                np.linspace(start_time, end_time, 5))
    plt.xlabel('Time (s)')
    plt.tight_layout()

    ax2 = plt.subplot(2,1,2, sharex=ax1, sharey=ax1)
    plt.plot(test, label='Signal')
    plt.vlines(test_beats, -2, 2, alpha=.5, color='g',
                linestyle='solid',linewidth=3, label='Beats')
    plt.legend(frameon=True, framealpha=0.75)
    # Limit the plot to a X-second window
    plt.xlim([start_time * sr, end_time * sr])
    plt.xticks(np.linspace(start_time, end_time, 5) * sr,
                np.linspace(start_time, end_time, 5))
    plt.xlabel('Time (s)')
    plt.tight_layout()
    plt.subplots_adjust(hspace=0)
    plt.show()
    return beat_score


# In[30]:

def call_beat_score(original_signal, test_signal, hop_size, start_time, end_time, sr, offbeat_factor):
    y, sr = librosa.load(test_signal, sr)
    onset_env = librosa.onset.onset_strength(y, sr=sr)
    tempo = librosa.beat.estimate_tempo(onset_env, sr=sr)
    score = analyze_signals(original_signal,test_signal, hop_size, start_time, end_time, sr, tempo, offbeat_factor)
    return score


# In[32]:

# example call
#call_beat_score('iknewyouweretrouble.wav','iknewyouweretrouble2.wav',512, 0, 30, 44100, .5)


# In[ ]:



