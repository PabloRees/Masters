import textblob
import pandas as pd
import nltk
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA
import json

vaderSentiment = SIA()

df = pd.read_csv('/Users/pablo/Desktop/Masters/Raw_speech_data/convention-speeches(60, 6).tsv',sep='\t')
transcriptsDict = {'Transcript':df['Transcript'],'No Stops Transcript':df['No Stops Transcript']}

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

sentDict = getSentiment(transcriptsDict)

df['BlobTranscriptSent'] = sentDict['BlobTranscriptSent']
df['BlobNoStopsSent'] = sentDict['BlobNoStopsSent']
df['VaderNoStopSent'] = sentDict['VaderNoStopSent']
df['VaderTranscriptSent'] = sentDict['VaderTranscriptSent']

print(df.columns)
print(df.loc[[0]])
print(df.iloc[0]['BlobNoStopsSent'])
print(df.iloc[0]['VaderTranscriptSent'])


