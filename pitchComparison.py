import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
import math
from noteAlignment import generateYAxis

# Pitch comparison function
#	Input: original song and singer's version's: array of fundamental frequencies, <==output of note separation function
#          array of note alignment indices <====== Output of noteAlignment function
#	Output: a rating,
#           subrating for each timestamp
#  Method: We will shift the F0 of the singer up/down an octave if they are half an octave or more off in pitch.
#  Then, create a triangular filter for how off the singer is relative to original version.

def pitchComparison(oriF0Array,singerF0Array,bestCorrespondingOriIndexList):

    lenSinger=len(singerF0Array)

    # turn F0 array into log F0 array
    # have a max here to prevent log(-1) gives us no number.
    oriLogF0Array=np.fmax(0,np.log2(oriF0Array))
    singerLogF0Array=np.fmax(0,np.log2(singerF0Array))

    subrating=[None for i in range(lenSinger)]
    rating=0.0
    numRated=0

    for i in range(lenSinger):
        if bestCorrespondingOriIndexList[i]!=None and singerLogF0Array[i]!=0.0:
            # check whether they're off by more than half an octave.
            # If so, move it up/down for them to stay within same octave
            currSingerLogF0=singerLogF0Array[i]
            currOriLogF0=oriLogF0Array[bestCorrespondingOriIndexList[i]]
            currSingerLogF0=currSingerLogF0+math.floor(currOriLogF0-currSingerLogF0+0.5)

            # triangle filter. Notes that are perfectly on pitch will have score 1.
            # Notes that are half-octave off will have score 0. Everything in between is scored linearly.
            subrating[i]=(1.0-abs(currOriLogF0-currSingerLogF0)/0.5)*100 # *100 to make everything on an 100 scale
            rating+=subrating[i]
            numRated+=1
        else:
            continue # The subrating will be None for notes that do not have correspondence.

    # divide by number of scores to get average scoring
    rating/=numRated


    return rating,subrating

#Note alignment function:
#	Input: original song and singer's version's:
#          array of fundamental frequencies, array of time stamps<== the output of note separation
#          array of index that matches singer notes to original notes.
#	Output: None. Just a graph
def visualize(oriF0Array,oriOffsetArray,singerF0Array,singerOffsetArray,rating,subrating):
    # First take the log
    oriLogF0Array=np.fmax(0,np.log2(oriF0Array))
    singerLogF0Array=np.fmax(0,np.log2(singerF0Array))

    lenOri=len(oriLogF0Array)
    lenSinger=len(singerLogF0Array)
    # Next we generate labels for each audio


    for oriIndex,oriLogF0 in enumerate(oriLogF0Array):
        if oriLogF0Array[oriIndex]!=0 and oriIndex+1<lenOri:
            # draw a horizontal line
            plt.plot([oriOffsetArray[oriIndex],oriOffsetArray[oriIndex+1]],[oriLogF0,oriLogF0],linewidth=5,c='r')


    for singerIndex,singerLogF0 in enumerate(singerLogF0Array):
        if singerLogF0Array[singerIndex]!=0 and singerIndex+1<lenSinger:
            # draw a horizontal line
            plt.plot([singerOffsetArray[singerIndex],singerOffsetArray[singerIndex+1]],[singerLogF0,singerLogF0],linewidth=5,c='b')

            if subrating[singerIndex]!=None:
                plt.annotate(
                    "{0:.1f}".format(subrating[singerIndex]),# Times 100 to make it look nicer.
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

    plt.text(xlim[1]-4, ylim[0]+0.5, 'Pitch rating: '+"{0:.1f}".format(rating), style='italic',
        bbox={'facecolor':'red', 'alpha':0.5, 'pad':10})

    plt.show()