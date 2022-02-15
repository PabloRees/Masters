import textblob
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import os
import json

vaderSentiment = SIA()


def getSentiment(transcriptsDict):
    BlobSentNoStops = []
    BlobSentTranscript = []
    vaderTranscriptSent = []
    vaderNoStopsSent = []

    print(f"Length = {len(transcriptsDict['Transcript'])}")
    for i in range(len(transcriptsDict['Transcript'])):
        print(i)

        transcriptBlob = textblob.TextBlob(transcriptsDict['Transcript'][i])
        noStopsBlob = textblob.TextBlob(transcriptsDict['No Stops Transcript'][i])

        BlobSentNoStops.append(transcriptBlob.sentiment)
        BlobSentTranscript.append(noStopsBlob.sentiment)

        myDict = ["'",",","    ","   ","  "]

        transcriptString = transcriptsDict['Transcript'][i]
        noStopsString = transcriptsDict['No Stops Transcript'][i]

        for s in myDict:
            transcriptString = transcriptString.replace(s,' ')
            noStopsString = noStopsString.replace(s,' ')

        vaderTranscriptSent.append(vaderSentiment.polarity_scores(transcriptString))
        vaderNoStopsSent.append(vaderSentiment.polarity_scores(noStopsString))

    transcriptsSentimentDict = {'BlobTranscriptSent':BlobSentTranscript,'BlobNoStopsSent':BlobSentTranscript,'VaderTranscriptSent':vaderTranscriptSent, 'VaderNoStopSent':vaderNoStopsSent}

    return transcriptsSentimentDict

def runSentAnal(dataFilePath,dataFilePath_save):
    for i in os.listdir(dataFilePath):
        df = pd.read_csv(dataFilePath + '/' + i, sep='\t')
        transcriptsDict = {'Transcript': df['Transcript'], 'No Stops Transcript': df['No Stops Transcript']}
        sentDict = getSentiment(transcriptsDict)

        df['BlobTranscriptSent'] = sentDict['BlobTranscriptSent']
        df['BlobNoStopsSent'] = sentDict['BlobNoStopsSent']
        df['VaderNoStopSent'] = sentDict['VaderNoStopSent']
        df['VaderTranscriptSent'] = sentDict['VaderTranscriptSent']

        df.to_csv(dataFilePath_save + '/' + i, sep='\t')

print(df.columns)
print(df.loc[[0]])
print(df.iloc[0]['BlobNoStopsSent'])
print(df.iloc[0]['VaderTranscriptSent'])


