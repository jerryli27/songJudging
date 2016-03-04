from YIN import Yin
import numpy
import scipy.io.wavfile
import segmenterModified
import matplotlib.pyplot as plt

sampleRate, monoAudioBuffer = scipy.io.wavfile.read('vaiueo2d.wav')#  Owl_City-Fireflies_Acapella_Official

freqs, offsets=segmenterModified.segmenter(monoAudioBuffer, sampleRate)
plt.plot(offsets,freqs)
plt.show()
print freqs
print offsets