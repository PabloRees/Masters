import pandas as pd

def transposeWordVectors(full_df,columnName):
    vectorsList = []
    for i in full_df[columnName]:
        vector = i.replace('[ ', '')
        vector = vector.replace('[', '')
        vector = vector.replace('\n', '')
        vector = vector.replace(']', '')
        vector = vector.replace('  ', ' ')

        vector = vector.split(' ')
        fVector = []
        for j in vector:
            if not j == '':
                fVector.append(float(j))

        vectorsList.append(fVector)

    vecNames = []
    for i in range(0, 200):
        vecNames.append('WV_' + str(i))

    vec_df = pd.DataFrame(vectorsList, columns=vecNames)

    full_df.drop(columnName,inplace=True, axis = 1)
    full_df.drop('X',inplace=True, axis = 1)
    full_df.drop('Unnamed..0',inplace=True, axis = 1)


    df_all_cols = pd.concat([full_df, vec_df], axis=1)

    return df_all_cols

def transposeSentiments(full_df):
    #negList = []
    #neuList = []
    #posList = []
    #compoundList = []
    vectorsList = []
    for i in full_df['vaderSent']:
        vec  = i.replace('{','')
        vec = vec.replace('}','')
        vec = vec.replace("'",'')
        vec = vec.split(', ')
        vector = [vec[0].split(' ')[-1],vec[1].split(' ')[-1],vec[2].split(' ')[-1],vec[3].split(' ')[-1]]
        vectorsList.append(vector)
        #negList.append(vec[0].split(' ')[-1])
        #neuList.append(vec[1].split(' ')[-1])
        #posList.append(vec[2].split(' ')[-1])
        #compoundList.append(vec[3].split(' ')[-1])

    #vectorsList = [negList,neuList,posList,compoundList]
    vader_df = pd.DataFrame(vectorsList, columns=['VaderNeg', 'VaderNeu', 'VaderPos', 'VaderComp'])

    vectorsList = []

    for i in full_df['blobSent']:

        vec = i.replace('Sentiment(polarity=','')
        vec = vec.replace(' subjectivity=','')
        vec = vec.replace(')','')
        vec = vec.split(',')
        vectorsList.append(vec)

    full_df.drop('vaderSent',inplace=True, axis = 1)
    full_df.drop('blobSent',inplace=True, axis = 1)

    sentBlob_df = pd.DataFrame(vectorsList, columns=['blobPol', 'blobSubj'])
    sent_df = pd.concat([vader_df, sentBlob_df], axis=1)
    df_all_cols = pd.concat([full_df, sent_df], axis=1)

    return df_all_cols

def transposeDocVectors(full_df,columnName,vecLen):


    vectorsList = []
    for i in full_df[columnName]:
        vector = i.replace('[ ', '')
        vector = vector.replace('[', '')
        vector = vector.replace('\n', '')
        vector = vector.replace(']', '')
        vector = vector.replace('  ', ' ')

        vector = vector.split(' ')


        fVector = []
        for j in vector:
            if not j == '':
                fVector.append(float(j))

        vectorsList.append(fVector)

    vecNames = []
    for i in range(0, vecLen):
        vecNames.append(f'DV_{str(vecLen)}_{str(i)}')

    vec_df = pd.DataFrame(vectorsList, columns=vecNames)

    full_df.drop(columnName,inplace=True, axis = 1)

    df_all_cols = pd.concat([full_df, vec_df], axis=1)

    return df_all_cols















