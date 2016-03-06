# Note: This is a modified version of segmenter.py. It does not contain non-note segment in the returned list. Previously the non-note part was represented as time=0 and freq=-1.


# import python pitch tracker
# now 'porting' praat.

import numpy
from YIN import Yin


def segmenter(y, fs):
    # frq_path, autoCorr_path, time, amp = praat_pd(y, fs)

    ytracker = Yin(fs)
    frq_path, amp = ytracker.trackPitch(y)

    freqs = frq_path[0]
    time = frq_path[1]
    # amps = amp / numpy.median(amp)
    # Normalize the amp values to be between 0 and 1
    amps = (amp - min(amp)) / (max(amp) - min(amp))
    # print amps
    N = len(freqs)

    # strict freq threshold
    # posHalfStep = pow(2, (1/12.))
    # negHalfStep = pow(2, (-1/12.))

    # More lenient freq threshold that identifies more note changes.
    # Needed for real-life instrument recordings lacking perfect pitch.
    posHalfStep = pow(2, (50/1200.))
    negHalfStep = pow(2, (-50/1200.))

    # The offset vector contains the times of the offsets where a new note
    # begins, and 0s elsewhere.
    noteFreqs=numpy.array([])
    offsets = numpy.array([])

    for k in range(1, N):
        # If two consecutive frequencies vary by more than a half step,
        # enter their time offset information into the offsets vector.
        if (freqs[k] / freqs[k-1] >= posHalfStep) or (freqs[k] / freqs[k-1] <= negHalfStep) or (amps[k] - amps[k-1] > 0.1):
            offsets=numpy.append(offsets,time[k])
            noteFreqs=numpy.append(noteFreqs,freqs[k])
    # reshape to 1d
    # offsets=numpy.reshape(offsets,(1,numpy.product(offsets.shape)))
    # noteFreqs=numpy.reshape(noteFreqs,(1,numpy.product(noteFreqs.shape)))
    offsets=offsets.flatten()
    noteFreqs=noteFreqs.flatten()
    return noteFreqs, offsets