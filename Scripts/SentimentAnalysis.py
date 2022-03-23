import textblob
from nltk.sentiment.vader import SentimentIntensityAnalyzer as SIA

def getSentiment(transcriptsList):
    vaderSentiment = SIA()
    blobSentList = []
    vaderSentList = []

    myDict = ["'", ",", "    ", "   ", "  "]

    for i in transcriptsList:

        blob = textblob.TextBlob(i)

        blobSentList.append(blob.sentiment)

        transcriptString = i

        for s in myDict:
            transcriptString = transcriptString.replace(s,' ')

        vaderSentList.append(vaderSentiment.polarity_scores(transcriptString))

    entimentDict = {'blobSent':blobSentList,'vaderSent':vaderSentList}

    return entimentDict




