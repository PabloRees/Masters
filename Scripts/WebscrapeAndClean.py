import requests
from bs4 import BeautifulSoup, SoupStrainer
import pandas as pd
import concurrent.futures
import time
from itertools import repeat
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
nltk_stopwords = set(stopwords.words('english'))
import lxml

print('Start')

#populates "results" with all the subcategories under "presidential", "press/media", "elections and transitions", "miscellaneous" - ignores "congressional"
class getBroadCategories:
    URL = 'https://www.presidency.ucsb.edu/documents'
    page = requests.get(URL)
    soup = BeautifulSoup(page.content, 'lxml')
    results = soup.find_all('ul', class_= 'dropdown-menu')
    print('Broad Categories End')

category = ''

#populates a list named "presidential" with extensions for all the subcategories in presidential
class presidential:
    presidentialS1 = str(getBroadCategories.results[1]).split('href="')
    for i in range(1): #the script tends to break because of gaps in internet connectivity - change the range to the number of categories already saved at the file destination to skip those subcategories
        presidentialS1.pop(0)

    presidential = []
    for i in presidentialS1:
        presidential.append(i.split('"')[0])

#populates a list named "press" with extensions for all the subcategories in press
class press:
    category = 'press'
    print('Start')
    pressS1 = str(getBroadCategories.results[2]).split('href="')
    pressS1.pop(0)
    press = []
    for i in pressS1:
        press.append(i.split('"')[0])

#populates a list named "elections" with extensions for all the subcategories in elections
class elections:
    electionsS1 = str(getBroadCategories.results[3]).split('href="')
    electionsS1.pop(0)
    elections = []
    for i in electionsS1:
      elections.append(i.split('"')[0])

#populates a list named "misc" with extensions for all the subcategories in miscellaneous
class misc:
    miscS1 = str(getBroadCategories.results[4]).split('href="')
    miscS1.pop(0)
    misc = []
    for i in miscS1:
        misc.append(i.split('"')[0])

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
    date = dateS2.split('" ', 1)[0]

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


#opens 60 links per page and saves the links into a list, then multithreads through the list using download_url
def multiThreading(numLinks, pageNum):
    transcriptUrls = []

    URL2 = URL + '?items_per_page=' + str(numLinks) +'&page=' + str(pageNum)
    page = requests.get(URL2)
    soup = BeautifulSoup(page.content, 'lxml')
    results = soup.find_all('div', class_='field-title')

    for j in results:
        urlS1 = str(j).split('href="', 1)[-1]
        urlS2 = urlS1.split('"', 1)[0]
        transcriptUrls.append('https://www.presidency.ucsb.edu' + urlS2)

    MAX_THREADS = 30
    length = len(transcriptUrls)

    if length == 0: length = 1

    threads = min(MAX_THREADS, length)

    with concurrent.futures.ThreadPoolExecutor(
            max_workers=threads) as executor:  # multithreading - its like 17 times faster than looping
        executor.map(download_url, transcriptUrls)

catsList = presidential.presidential + press.press + elections.elections + misc.misc


for i in catsList:
    print(i)
    t0 = time.time()
    URL = 'https://www.presidency.ucsb.edu' + i           #1
    page = requests.get(URL)                              #2
    soup = BeautifulSoup(page.content, 'lxml', parse_only=SoupStrainer('div', class_='tax-count'))  #3
    results = soup.find('div', class_='tax-count')        #4: these 6 lines gets the number of documents in each sub category of "presidential"
    numDocsS1 = str(results).split(' of ')[-1]            #5
    numDocs = int(numDocsS1.split('.')[0])                #6
    print(numDocs, " transcripts to be scraped")

    if numDocs > 59:
        fraction = round((numDocs/60)-0.5)
    else:
        fraction = 1

    namesList=[]
    datesList = []
    titlesList = []
    transcriptsList=[]
    noStopsTranscriptsList=[]


    pageNum = 0
    t0 = time.time()
    print(f'Fraction = {fraction}')
    for p in range(fraction):
        t2 = time.time()
        multiThreading(60, pageNum)
        t3 = time.time()
        print("60 done in ",t3-t2," seconds = ", 60/(t3-t2)," every second = 1 every ", (t3-t2)/60, " seconds." )
        pageNum+=1
    t3 = time.time()

    typeList = []
    typeList.extend(repeat(i.split('app-categories/',1)[-1],len(namesList)))
    print(typeList)


    dict = {'Type':typeList,'Name':namesList,'Date':datesList, 'Title':titlesList, 'Transcript':transcriptsList, 'No Stops Transcript':noStopsTranscriptsList}
    df = pd.DataFrame(dict)

    df['Date'] = pd.to_datetime(df['Date'])
    df.sort_values(by='Date', inplace=True, ascending=False)

    df2 = df[['Type','Name','Date','Title','Transcript','No Stops Transcript']].copy()

    dataShape = str(df2.shape)

    filePath = '/Users/pablo/Desktop/Masters /Raw_speech_data/'

    fileName = i.split('/')[-1] + dataShape

    df2.to_csv(filePath+fileName+ '.tsv',sep = '\t', index=False) #saving the data frame as a .tsv file

    df_partial = df2.iloc[:10, :]
    df_partial.to_csv(filePath + fileName + '_sample.tsv')

    t1=time.time()
    print('\n_________________________________________________________________________________________\n')
    print(numDocs, " transcripts in ", round(t1-t0), "seconds")
    print('\n_________________________________________________________________________________________\n')
