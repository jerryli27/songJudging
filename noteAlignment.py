import numpy as np
import matplotlib.pyplot as plt
import math
import matplotlib.lines as mlines


LOG_FO_WINDOW=0.20
OFFSET_WINDOW=1.0
VITERBI_WINDOW=4

storedMaxProbPath={}

#Note alignment function:
#	Input: original song and singer's version's:
#          array of fundamental frequencies, array of time stamps<== the output of note separation
#	Output: an alignment, that is an array of index for the singer version's notes where the index is
#            the corresponding note in the original version. [0,1,1,2] means that first note in singer version
#            correspond to first note in original version, second note correspond to second note,
#            third note in singer version correspond to the second note in original version,
#            and fourth note in singer version correspond to original version.
def noteAlignment(oriF0Array,oriOffsetArray,singerF0Array,singerOffsetArray):
    storedMaxProbPath.clear()
    lenOri=len(oriOffsetArray)
    lenSinger=len(singerOffsetArray)

    # turn F0 array into log F0 array
    # have a max here to prevent log(-1) gives us no number.
    oriLogF0Array=np.fmax(0,np.log2(oriF0Array))
    singerLogF0Array=np.fmax(0,np.log2(singerF0Array))

    # create a list of candidates for each note.
    candidates=[[] for i in range(lenSinger)]

    # for each note in the singer,, find the notes in the original song that is within the offset window.
    oriNoteStartIndex=0
    oriNoteEndIndex=0
    i=0
    while i<lenSinger:
        # find the notes that is contained within the window
        while (oriNoteStartIndex<lenOri) and (oriOffsetArray[oriNoteStartIndex]<singerOffsetArray[i]-OFFSET_WINDOW):
            oriNoteStartIndex+=1
            # This is the index of the first note inside the window
        while (oriNoteEndIndex<lenOri) and (oriOffsetArray[oriNoteEndIndex]<singerOffsetArray[i]+OFFSET_WINDOW):
            oriNoteEndIndex+=1
            # Note, the oriNoteEndIndex is where the note is just outside the offset window, not inside the window.
        candidates[i]= range(oriNoteStartIndex,oriNoteEndIndex)
        i+=1

    # now we calculate the probability that each singer's note align with the candidates (the notes in original song)
    candidatesProbList=[[probNoteAlign(oriLogF0Array[j],oriOffsetArray[j],singerLogF0Array[i],singerOffsetArray[i]) for j in candidates[i]] for i in range(lenSinger)]

    # We use HMM and Viterbi algorithm to calculate the most probable path.
    # bestCorrespondingOriIndexList stores (probability, bestOriIndex) for each note.
    bestCorrespondingOriIndexList=[(0,None) for i in range(lenSinger)]
    for singerIndex in range(lenSinger):
        bestCorrespondingOriIndexList[singerIndex]=viterbiDP(VITERBI_WINDOW,candidates,candidatesProbList,singerIndex,bestCorrespondingOriIndexList[max(singerIndex-1,0)][1])
        if singerIndex % 100 == 0:
            print(str(singerIndex + 1) + ' of ' + str(lenSinger) + ' hidden state completed viterbi best candidate calculation')
    # Right now we're not returning the probability of each note correspondence.
    # If we want, we can return bestCorrespondingOriIndexList

    return [i[1] for i in bestCorrespondingOriIndexList]

# Desc: given two notes, one in the original song and the other in the singer's version,
# this calculates the probability that the singer's note aligns with the note in the original song.
# Input: log of fundamental frequency and time offset of both original and singer's note.
# Output: probability that they align (double)
def probNoteAlign(oriNoteLogF0,oriNoteOffset,singerNoteLogF0,singerNoteOffset):
    # Here we just use a trangular probability. If original F0 is exactly singer F0, then prob=1
    # Prob decrease to near 0 when oriNoteLogF0-singerNoteLogF0=LOG_FO_WINDOW
    # same thing for note offset.

    # To fix the bug that -1 tries its best to align with other -1s, we modify the probability.
    # IT is possible that a -1 aligns with any other note in the original song, as long as it's in the offset window.
    if singerNoteLogF0==0:
        # if it is the end of a note
        if oriNoteLogF0==0:
            # end of a note aligns with end of a note
            return (1-abs(oriNoteOffset-singerNoteOffset)/OFFSET_WINDOW)
        else:
            # end of a note aligns with any other kinds of note.
            return 0.5*(1-abs(oriNoteOffset-singerNoteOffset)/OFFSET_WINDOW)
    else:
        return (1-abs(oriNoteLogF0-singerNoteLogF0)/LOG_FO_WINDOW)*(1-abs(oriNoteOffset-singerNoteOffset)/OFFSET_WINDOW)



# Desc: given two notes in the original song, this calculates the probability that one transition to the other
# Input: index of first note and second note
# Output: probability (double)
def probTransition(oriNoteCurrIndex,oriNoteNextIndex):
    if oriNoteCurrIndex==None:
        # If there is no previous note, then we can start anywhere. The prob is 1.
        return 1.0
    elif oriNoteNextIndex==oriNoteCurrIndex+1:
        # If the next index is curr index + 1, that means it's probably a normal transition. Therefore it's probable
        return 0.5
    elif oriNoteNextIndex==oriNoteCurrIndex:
        # If the next index is curr index, that means two singer's note correspond to the same original note
        # This is less probable
        return 0.25
    elif oriNoteNextIndex>oriNoteCurrIndex+1:
        # If the next index is more than curr index+1 , that means we skipped one original note
        # This is even less probable
        return 0.2
    else:
        # If the next index is less than curr index , that means we 're going back to a previous note.
        # This is almost impossible.
        return 0.05
    
# Input: windowLength, candidatesList (a list, candidatesList[i][j]=the jth hidden state candidate for observed state i.
# candidatesProbList (a list, candidatesProbList[i][j]= prob of observed i is hidden state candidate j)),
# observationStartIndex: which observed state are we currently processing.
# lastHiddenStateIndex: what is the last hidden state? Used to calculate transition prob.
# Output: prob,bestHiddenStateIndex
def viterbi(windowLength,candidatesList,candidatesProbList,observationStartIndex,lastHiddenStateIndex):
    ret=-1
    bestHiddenStateIndex=None
    tup=(windowLength,observationStartIndex,lastHiddenStateIndex)
    # if storedMaxProbPath.get(tup,None)!=None:
    #     # if we've calculated this before, get that value
    #     ret,bestHiddenStateIndex=storedMaxProbPath.get(tup,None)
    #     return ret,bestHiddenStateIndex
    if windowLength==0 or len(candidatesProbList)<=observationStartIndex:
        # If window=0 or if we've reached the end of the candidatesProbList.
        return ret,bestHiddenStateIndex
    elif windowLength==1:
        for i in range(len(candidatesProbList[observationStartIndex])):
            currHiddenIndex=candidatesList[observationStartIndex][i]
            probObIsHidden=candidatesProbList[observationStartIndex][i]
            if ret<probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex):
                ret=probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex)
                bestHiddenStateIndex=currHiddenIndex
        # store the calculated result
        #storedMaxProbPath[tup]=(ret,bestHiddenStateIndex)
        return ret,bestHiddenStateIndex
    else:
        ret=-1
        bestHiddenStateIndex=None
        for i in range(len(candidatesProbList[observationStartIndex])):
            currHiddenIndex=candidatesList[observationStartIndex][i]
            probObIsHidden=candidatesProbList[observationStartIndex][i]
            nextBestProb,nextBestHiddenStateIndex=viterbi(windowLength-1,candidatesList,candidatesProbList,observationStartIndex+1,currHiddenIndex)
            if nextBestHiddenStateIndex==None:
                # If for some reason we cannot find the next best road, then treat this case as if our window size is 1
                if ret<probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex):
                    ret=probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex)
                    bestHiddenStateIndex=currHiddenIndex
            else:
                # Record the best route so far.
                if ret<probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex)*nextBestProb:
                    ret=probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex)*nextBestProb
                    bestHiddenStateIndex=currHiddenIndex
        # store the calculated result
        #storedMaxProbPath[tup]=(ret,bestHiddenStateIndex)
        return ret,bestHiddenStateIndex
    

def viterbiDP(windowLength,candidatesList,candidatesProbList,observationStartIndex,lastHiddenStateIndex):
    ret=-1
    bestHiddenStateIndex=None
    tup=(windowLength,observationStartIndex,lastHiddenStateIndex)
    if storedMaxProbPath.get(tup,None)!=None:
        # if we've calculated this before, get that value
        ret,bestHiddenStateIndex=storedMaxProbPath.get(tup,None)
        return ret,bestHiddenStateIndex
    if windowLength==0 or len(candidatesProbList)<=observationStartIndex:
        # If window=0 or if we've reached the end of the candidatesProbList.
        return ret,bestHiddenStateIndex
    elif windowLength==1:
        for i in range(len(candidatesProbList[observationStartIndex])):
            currHiddenIndex=candidatesList[observationStartIndex][i]
            probObIsHidden=candidatesProbList[observationStartIndex][i]
            if ret<probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex):
                ret=probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex)
                bestHiddenStateIndex=currHiddenIndex
        # store the calculated result
        storedMaxProbPath[tup]=(ret,bestHiddenStateIndex)
        return ret,bestHiddenStateIndex
    else:
        ret=-1
        bestHiddenStateIndex=None
        for i in range(len(candidatesProbList[observationStartIndex])):
            currHiddenIndex=candidatesList[observationStartIndex][i]
            probObIsHidden=candidatesProbList[observationStartIndex][i]
            nextBestProb,nextBestHiddenStateIndex=viterbiDP(windowLength-1,candidatesList,candidatesProbList,observationStartIndex+1,currHiddenIndex)
            if nextBestHiddenStateIndex==None:
                # If for some reason we cannot find the next best road, then treat this case as if our window size is 1
                if ret<probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex):
                    ret=probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex)
                    bestHiddenStateIndex=currHiddenIndex
            else:
                # Record the best route so far.
                if ret<probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex)*nextBestProb:
                    ret=probObIsHidden*probTransition(lastHiddenStateIndex,currHiddenIndex)*nextBestProb
                    bestHiddenStateIndex=currHiddenIndex
        # store the calculated result
        storedMaxProbPath[tup]=(ret,bestHiddenStateIndex)
        return ret,bestHiddenStateIndex

#Note alignment function:
#	Input: original song and singer's version's:
#          array of fundamental frequencies, array of time stamps<== the output of note separation
#          array of index that matches singer notes to original notes.
#	Output: None. Just a graph
def visualize(oriF0Array,oriOffsetArray,singerF0Array,singerOffsetArray,bestCorrespondingOriIndexList):
    # First take the log
    oriLogF0Array=np.fmax(0,np.log2(oriF0Array))
    singerLogF0Array=np.fmax(0,np.log2(singerF0Array))

    lenOri=len(oriLogF0Array)
    lenSinger=len(singerLogF0Array)

    # Next we generate labels for each audio
    oriLabel=[str(i) for i in range(lenOri)]
    singerLabel=[str(i) for i in bestCorrespondingOriIndexList]

    for oriIndex,oriLogF0 in enumerate(oriLogF0Array):
        if oriLogF0Array[oriIndex]!=0 and oriIndex+1<lenOri:
            # draw a horizontal line
            plt.plot([oriOffsetArray[oriIndex],oriOffsetArray[oriIndex+1]],[oriLogF0,oriLogF0],linewidth=5,c='r')
            if oriIndex in bestCorrespondingOriIndexList:
                plt.annotate(
                    oriLabel[oriIndex],
                    xy = ((oriOffsetArray[oriIndex]+oriOffsetArray[oriIndex+1])/2.0, oriLogF0), xytext = (-20, -20),
                    textcoords = 'offset points', ha = 'right', va = 'bottom',
                    bbox = dict(boxstyle = 'round,pad=0.5', fc = 'red', alpha = 0.5),
                    arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))

    # singerAx = fig.add_subplot(212)

    for singerIndex,singerLogF0 in enumerate(singerLogF0Array):
        if singerLogF0Array[singerIndex]!=0 and singerIndex+1<lenSinger:
            # draw a horizontal line
            plt.plot([singerOffsetArray[singerIndex],singerOffsetArray[singerIndex+1]],[singerLogF0,singerLogF0],linewidth=5,c='b')
            plt.annotate(
                singerLabel[singerIndex],
                xy = ((singerOffsetArray[singerIndex]+singerOffsetArray[singerIndex+1])/2.0, singerLogF0), xytext = (-20, 20),
                textcoords = 'offset points', ha = 'right', va = 'bottom',
                bbox = dict(boxstyle = 'round,pad=0.5', fc = 'blue', alpha = 0.5),
                arrowprops = dict(arrowstyle = '->', connectionstyle = 'arc3,rad=0'))


    maxLogF0=max(np.hstack((oriLogF0Array,singerLogF0Array)))
    minLogF0Without0=maxLogF0 # initialize to a large number first.

    for i in np.hstack((oriLogF0Array,singerLogF0Array)):
        if i>0:
            minLogF0Without0=min(minLogF0Without0,i)

    xlim=[min(np.hstack((oriOffsetArray,singerOffsetArray))),max(np.hstack((oriOffsetArray,singerOffsetArray)))]
    ylim=[minLogF0Without0-0.5,maxLogF0+0.5]  # -4 and +4 to make drawing nicer.

    yticks=generateYAxis(ylim)
    xrange=np.arange(start=0,stop=xlim[1]+0.05,step=0.2) # step 0.2 to make it look nicer
    yrange=np.arange(start=ylim[0],stop=ylim[1],step=(ylim[1]-ylim[0])/(np.floor((ylim[1]-ylim[0])*12.0)))
    plt.xticks(xrange)
    plt.yticks(yrange,yticks)
    plt.xlabel('Time(s)')
    plt.ylabel('Notes(Western Notations)')

    # Play a little trick on the legend to make it show the right thing.
    oriLegend = mlines.Line2D([], [], linewidth=5,c='r',label='Original')
    singerLegend = mlines.Line2D([], [], linewidth=5,c='b',label='Singer')
    plt.legend(handles=[oriLegend,singerLegend])

    plt.show()

# input the limit of y.
# output the ticks for y axis
def generateYAxis(ylim):
    C4log=np.log2(440.0*2.0**(-9.0/12.0))
    pianoKeys=['G#','A','A#','B','C','C#','D','D#','E','F','F#','G'] # The twelve piano keys

    lowestKey=math.floor((ylim[0]-C4log)*12)+40.0
    highestKey=math.floor((ylim[1]-C4log)*12)+40.0

    # lowestKey=pianoKeys[math.floor((ylim[0]-C4log)*12)%12]+str(math.floor(ylim[0]-C4log)+4)
    # highestKey=pianoKeys[math.floor((ylim[1]-C4log)*12)%12]+str(math.floor(ylim[1]-C4log)+4)
    yticks=[pianoKeys[int(i%12)]+str(int(math.floor((i-40)/12.0)+4))  for i in range(int(lowestKey),int(highestKey))]
    return yticks

