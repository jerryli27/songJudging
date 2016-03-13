# with open("noteAlignment.py") as fp:
#     for i, line in enumerate(fp):
#         if "\xe2" in line:
#             print i, repr(line)
#

from YIN import Yin
import numpy
import scipy.io.wavfile
import matplotlib.pyplot as plt
import random
import timeit
import time
import librosa

import segmenterModified
import noteAlignment
import pitchComparison
import BeatTracker

# #  Owl_City-Fireflies_Acapella_Official.wav vaiueo2d.wav
# sampleRate, monoAudioBuffer = scipy.io.wavfile.read('Owl_City-Fireflies_Acapella_Official.wav')
#
# freqs, offsets=segmenterModified.segmenter(monoAudioBuffer, sampleRate)
# # plt.plot(offsets,freqs)
# # plt.show()
# print freqs
# print offsets


# Sanity check for note Alignment
# First just add some distortion to freq and offset. See if the model is robust to that.
# Add some distortion in the freqs and offsets.
# singerFreqs=freqs+numpy.array([random.uniform(-5,5) for i in range(len(freqs))])
# singerOffsets=offsets+numpy.array([random.uniform(-0.05,0.05) for i in range(len(offsets))])
# print singerFreqs
# print singerOffsets
# print noteAlignment.noteAlignment(freqs,offsets,singerFreqs,singerOffsets)


# Sanity check 2
# Then add duplicate notes and also distortion. See if the model is robust to that.
# singerFreqs=numpy.array([])
# for i in freqs:
#     singerFreqs=numpy.append(singerFreqs,i)
#     singerFreqs=numpy.append(singerFreqs,i)
# singerOffsets=numpy.array([])
# for i in offsets:
#     singerOffsets=numpy.append(singerOffsets,i)
#     singerOffsets=numpy.append(singerOffsets,i)
#
# singerFreqs=singerFreqs+numpy.array([random.uniform(-5,5) for i in range(len(freqs)*2)])
# singerOffsets=singerOffsets+numpy.array([random.uniform(0,0.025) for i in range(len(offsets)*2)])
# print singerFreqs
# print singerOffsets
# print noteAlignment.noteAlignment(freqs,offsets,singerFreqs,singerOffsets)


# *** Check how long it typically takes to run note alignment.
# startTime=time.time()
# numLoops=100
# for i in range(numLoops):
#     noteAlignment.noteAlignment(freqs,offsets,singerFreqs,singerOffsets)
# endTime=time.time()
# print 'The time it took is on average: '+str((endTime-startTime)/numLoops)+'s'


# Sanity check 3. Same version
#
# sampleRate, monoAudioBuffer = scipy.io.wavfile.read('Taylor_Swift_-_I_Knew_You_Were_Trouble_30s.wav')
# freqs, offsets=segmenterModified.segmenter(monoAudioBuffer, sampleRate)
# print freqs
# print offsets
#
#
# sampleRate, monoAudioBuffer = scipy.io.wavfile.read('Taylor_Swift_-_I_Knew_You_Were_Trouble_30s.wav')#'Me_Singing_I_Knew_You_Were_Trouble_30s.wav'
# singerFreqs,singerOffsets=segmenterModified.segmenter(monoAudioBuffer, sampleRate)
# print singerFreqs
# print singerOffsets
#
# bestCorrespondingOriIndexList=noteAlignment.noteAlignment(freqs,offsets,singerFreqs,singerOffsets)
# print bestCorrespondingOriIndexList
# #
# noteAlignment.visualize(freqs,offsets,singerFreqs,singerOffsets,bestCorrespondingOriIndexList)
#
# # Found out that the current algorithm tries to align -1 with -1 (Because otherwise it does not have any other candidates).
# # Maybe modify it so that -1 can align with nothing?
#
#
# pitchScore,pitchSubrating=pitchComparison.pitchComparison(freqs,singerFreqs,bestCorrespondingOriIndexList)
# pitchComparison.visualize(freqs,offsets,singerFreqs,singerOffsets,pitchScore,pitchSubrating)
#
# beatScore=BeatTracker.call_beat_score('Taylor_Swift_-_I_Knew_You_Were_Trouble_30s.wav','Taylor_Swift_-_I_Knew_You_Were_Trouble_30s.wav',512, 0, 30, 44100, .5)
# # 'Me_Singing_I_Knew_You_Were_Trouble_30s.wav'
# totalScore=(pitchScore*0.7+beatScore*0.3)
#
# print 'The beat score is: '+"{0:.1f}".format(beatScore)+'The pitch score is: '+"{0:.1f}".format(pitchScore)+\
#       'The final score is: '+"{0:.1f}".format(totalScore)

# Sanity check 4
# shift the pitch by 2 half steps


sampleRate, monoAudioBuffer = scipy.io.wavfile.read('Taylor_Swift_-_I_Knew_You_Were_Trouble_30s.wav')
freqs, offsets=segmenterModified.segmenter(monoAudioBuffer, sampleRate)
print freqs
print offsets


sampleRate, monoAudioBuffer = scipy.io.wavfile.read('Taylor_Swift_-_I_Knew_You_Were_Trouble_30s_2halfsteps.wav')#'Me_Singing_I_Knew_You_Were_Trouble_30s.wav'
singerFreqs,singerOffsets=segmenterModified.segmenter(monoAudioBuffer, sampleRate)
print singerFreqs
print singerOffsets

bestCorrespondingOriIndexList=noteAlignment.noteAlignment(freqs,offsets,singerFreqs,singerOffsets)
print bestCorrespondingOriIndexList
#
noteAlignment.visualize(freqs,offsets,singerFreqs,singerOffsets,bestCorrespondingOriIndexList)

# Found out that the current algorithm tries to align -1 with -1 (Because otherwise it does not have any other candidates).
# Maybe modify it so that -1 can align with nothing?


pitchScore,pitchSubrating=pitchComparison.pitchComparison(freqs,singerFreqs,bestCorrespondingOriIndexList)
pitchComparison.visualize(freqs,offsets,singerFreqs,singerOffsets,pitchScore,pitchSubrating)

beatScore=BeatTracker.call_beat_score('Taylor_Swift_-_I_Knew_You_Were_Trouble_30s.wav','Taylor_Swift_-_I_Knew_You_Were_Trouble_30s_2halfsteps.wav',512, 0, 30, 44100, .5)
# 'Me_Singing_I_Knew_You_Were_Trouble_30s.wav'
totalScore=(pitchScore*0.7+beatScore*0.3)

print 'The beat score is: '+"{0:.1f}".format(beatScore)+'The pitch score is: '+"{0:.1f}".format(pitchScore)+\
      'The final score is: '+"{0:.1f}".format(totalScore)

# Sanity check 5. Take two completely different songs and see what's the score.
#
# sampleRate, monoAudioBuffer = scipy.io.wavfile.read('Taylor_Swift_-_I_Knew_You_Were_Trouble_30s.wav')
# freqs, offsets=segmenterModified.segmenter(monoAudioBuffer, sampleRate)
# print freqs
# print offsets
#
#
# sampleRate, monoAudioBuffer = scipy.io.wavfile.read('fireflies_30s.wav')#'Me_Singing_I_Knew_You_Were_Trouble_30s.wav'
# singerFreqs,singerOffsets=segmenterModified.segmenter(monoAudioBuffer, sampleRate)
# print singerFreqs
# print singerOffsets
#
# bestCorrespondingOriIndexList=noteAlignment.noteAlignment(freqs,offsets,singerFreqs,singerOffsets)
# print bestCorrespondingOriIndexList
# #
# noteAlignment.visualize(freqs,offsets,singerFreqs,singerOffsets,bestCorrespondingOriIndexList)
#
# # Found out that the current algorithm tries to align -1 with -1 (Because otherwise it does not have any other candidates).
# # Maybe modify it so that -1 can align with nothing?
#
#
# pitchScore,pitchSubrating=pitchComparison.pitchComparison(freqs,singerFreqs,bestCorrespondingOriIndexList)
# pitchComparison.visualize(freqs,offsets,singerFreqs,singerOffsets,pitchScore,pitchSubrating)
#
# beatScore=BeatTracker.call_beat_score('Taylor_Swift_-_I_Knew_You_Were_Trouble_30s.wav','fireflies_30s.wav',512, 0, 30, 44100, .1)
# # 'Me_Singing_I_Knew_You_Were_Trouble_30s.wav'
# totalScore=(pitchScore*0.7+beatScore*0.3)
#
# print 'The beat score is: '+"{0:.1f}".format(beatScore)+'The pitch score is: '+"{0:.1f}".format(pitchScore)+\
#       'The final score is: '+"{0:.1f}".format(totalScore)
#
