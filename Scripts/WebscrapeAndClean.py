import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
import concurrent.futures
import time
from itertools import repeat
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words('english'))
import os

namesList=[]
datesList = []
titlesList = []
transcriptsList=[]
noStopsTranscriptsList=[]

#populates "results" with all the subcategories under "presidential", "press/media", "elections and transitions", "miscellaneous" - ignores "congressional"
def getBroadCategories():
    URL = 'https://www.presidency.ucsb.edu/documents'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'lxml')
    results = soup.find_all('ul', class_= 'dropdown-menu')
    print('Broad Categories End')

    #creates categories list from the presidency project website
    catsList = []
    for i in range(1,5):
        cat = str(results[i]).split('href="')
        for j in cat:
            catsList.append(j.split('"')[0])
    for i in range(0,26):
        catsList.pop(0)
    return(catsList)

#downloads a speech transcription and saves the transcript, speaker, date and title to a list
def download_url(url):
    page3 = requests.get(url)  # These 3 lines open the speech transcript
    soup3 = BeautifulSoup(page3.content, 'lxml', parse_only=SoupStrainer('section', class_="col-sm-9"))

    nameS1 = soup3.find('h3', class_="diet-title")
    nameS2 = str(nameS1).split('href="', 1)[-1]  # These 3 lines save the speakers title and name to 'name'
    name = nameS2.split('"', 1)[0]

    titleS1 = soup3.find('div', class_="field-ds-doc-title")
    titleS2 = str(titleS1).split('<h1>', 1)[-1]  # These 3 lines save the transcript title to 'title'
    title = titleS2.split('</h1>', 1)[0]

    dateS1 = soup3.find('span', class_="date-display-single")
    dateS2 = str(dateS1).split('content="', 1)[-1]  # These 3 lines save the transcript dateTime title to 'date'
    date3 = dateS2.split('" ', 1)[0]
    print(date3)
    date = pd.to_datetime(date3)

    transcriptS1 = soup3.find('div', class_="field-docs-content")
    transcriptS2 = str(transcriptS1).split('">', 1)[-1]  # These 3 lines save the transcript to 'transcript'
    transcript = transcriptS2.split('</div>', 1)[0]

    myDict = ['<p>','</p>','[<i>laughter</i>]','[<i>Laughter</i>]','[<em>Laughter</em>]','[<em>laughter</em>]','[<em>Applause</em>]','[<em>applause</em>]','[<i>Applause</i>]','[<i>applause</i>]','—', '<i>','<em>','</em>','<', '/i', '>','-',':',',','[ilaughter]','[iLaughter]','[i','[iApplause]','[emLaughter/em]','[em','/em]','em]','/em','/','[',']','"',"'",';']

    for s in myDict:
        transcript = transcript.replace(s,'') #remove markup characters

    if 'Q:' in transcript:
        transcript = (transcript.split('Q:', 1)[0]).lower()  #split the press brief from the following questions
    else: transcript = transcript.lower()

    transcript = word_tokenize(transcript) #tokenize
    noStopsTranscript = [] #create a variable for transcripts with stopwords removed

    for token in transcript:
        if token not in nltk_stopwords:
            noStopsTranscript.append(token) #clean stopwords out of noStopsTranscript

    namesList.append(name)
    transcriptsList.append(transcript)
    noStopsTranscriptsList.append((noStopsTranscript))
    datesList.append(date)
    titlesList.append(title)

#Multithreads through a list of URLs using download_url
def multiThreading(transcriptUrls):
    MAX_THREADS = 30
    length = len(transcriptUrls)
    if length == 0: length = 1

    threads = min(MAX_THREADS, length)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=threads) as executor:  # multithreading - its like 17 times faster than looping
        executor.map(download_url, transcriptUrls)

def WScrape(saveFilePath,category):
    URL = 'https://www.presidency.ucsb.edu' + category           #1
    page = requests.get(URL)                              #2
    soup = BeautifulSoup(page.content, 'lxml', parse_only=SoupStrainer('div', class_='tax-count'))  #3
    results = soup.find('div', class_='tax-count')        #4: these 6 lines gets the number of documents in each sub category of "presidential"
    numDocsS1 = str(results).split(' of ')[-1]            #5
    numDocs = int(numDocsS1.split('.')[0])                #6

    if numDocs > 59:
        fraction = round((numDocs/60)-0.49)
    else:
        fraction = 1

    pageNum = 0
    t0 = time.time()

    for page_num in range(fraction):
        t2 = time.time()

        # opens 60 links per page and saves the links into a list
        transcriptUrls = []
        URL2 = URL + '?items_per_page=' + str(60) + '&page=' + str(pageNum)
        page = requests.get(URL2)
        soup = BeautifulSoup(page.content, 'lxml')
        results = soup.find_all('div', class_='field-title')

        for j in results:
            urlS1 = str(j).split('href="', 1)[-1]
            urlS2 = urlS1.split('"', 1)[0]
            transcriptUrls.append('https://www.presidency.ucsb.edu' + urlS2)

        multiThreading(transcriptUrls)

        pageNum+=1

    typeList = []
    typeList.extend(repeat(category.split('app-categories/',1)[-1],len(namesList)))

    dict = {'Type':typeList,'Name':namesList,'Date':datesList, 'Title':titlesList, 'Transcript':transcriptsList, 'No Stops Transcript':noStopsTranscriptsList}
    df = pd.DataFrame(dict)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True, ascending=False)

    df2 = df[['Type','Name','Date','Title','Transcript','No Stops Transcript']].copy()

    dataShape = str(df2.shape)

    fileName = category.split('/')[-1] + dataShape

    df2.to_csv(saveFilePath + '/' + fileName + '.tsv',sep = '\t', index=False) #saving the data frame as a .tsv file

    #df_partial = df2.iloc[:10, :]
    #df_partial.to_csv(saveFilePath + '/' + fileName + '_sample.tsv')

    t1=time.time()

def runWSC(speechDataSavePath):
    catsList = getBroadCategories()
    print('looping')
    for i in catsList:
        if not i == '<ul class=':
            print(i)
            WScrape(speechDataSavePath, i)
            print('resetting global lists')
            namesList.clear()
            datesList.clear()
            titlesList.clear()
            transcriptsList.clear()
            noStopsTranscriptsList.clear()


#runWSC('/Users/pablo/Desktop/Masters/Github_Repository/Masters/Data/Speech_data_test')