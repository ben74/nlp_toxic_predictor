# -*-coding:utf-8 -*
#---♥✨🗲 ⚡ΨͶ ֍߷ओ
import gv
''' sets and gets "globals" at gv scope '''
def GG(k):
  return gv.a[k]
  
def SG(k,v):
  gv.a[k]=v     
SG('verbose',False)
  
import sklearn.preprocessing
import seaborn as sns
import matplotlib.pyplot as plt
import datetime
import requests
import re
import warnings
import pickle
import gc
import subprocess
import operator
import os
import sys
import json
import numpy as np
import pandas as pd
from time import time as tim
from scipy import stats
from webptools import webplib
import pysftp
import hashlib
from IPython.display import clear_output

echo=print
p=print
p('Last Run:',datetime.datetime.now())
#p('Mainframe alpow included')  # mainframe
# import importlib;importlib.reload(alpow);#reload
#os.system('rm -f alpow.py;wget https://alpow.fr/alpow.py')
# requirements

if('requiredModules' not in globals().keys()):
    requiredModules='wget joblib ipython sklearn seaborn Flask webptools pysftp numpy requests unidecode tensorflow'.split(' ')
fn='modules-versions.txt'
os.system('pip freeze > '+fn)
installed=''
with open(fn) as f:
    installed += f.read()
    
for module in requiredModules:
    if(module+'==' not in  installed):
        print('Trying to install :',module)
        os.system('pip install '+module)

# }conf{
np.random.seed(seed=23021983)
pd.set_option('display.max_rows', 900)
pd.set_option('display.max_columns', 40)
# or evaluate js document frame innerWidth .
pd.set_option('display.width', 1200)
plt.style.use('fivethirtyeight')
plt.rcParams["figure.figsize"] = (24, 12)
plt.rcParams['figure.facecolor'] = 'white'
#warnings.filterwarnings("ignore", category=DeprecationWarning)
warnings.filterwarnings('ignore')
cnopts = pysftp.CnOpts()
cnopts.hostkeys = None
lab_enc = sklearn.preprocessing.LabelEncoder()
# }vars{
heightPerGraph = widthPerGraph = 6
demo = 1
verbose = 0

#mieux de faire ceci
SG('sftp',{'cd': '', 'web': '-', 'h': '-', 'u': '-', 'p': '-'})    
SG('webRepo','')
SG('useFTP',True)
SG('ftplist',[])

#empty parameters to be overriden => au niveau des globales de ce module uniquement !!
def setG(k,v):
    if(k in globals().keys()):
        del(globals()[k]);
    globals()[k]=v
    
def g(x):
    return globals()[x]
    
considerEmpty = [np.inf, -np.inf, np.nan, 0, 'na', '']

sendimages2ftp=1
removepng=1
# }shortcuts{


# }functions{
def md5(x):
    res = hashlib.md5(str(x).encode())
    return res.hexdigest()


def show():
    global demo
    if(demo):
        plt.show()
        return
    plt.close()

    
def findWithinArrayValues(tags,q):
  res=[]
  for k in range(len(tags)):
  #for k,v in tags.iterows():
    v=tags[k]
    try:
      if(v.index(q)):
        res+=[k]
    except ValueError:
      pass
  return res

def message(x):
    now = datetime.datetime.now()
    url = 'https://1.x24.fr/a/bus.php'
    r = requests.post(url, data={0: x})
# re is not defined ... whut ???


def filename(x):
    return re.sub(r"[^a-z0-9\-_,\.:]+", '-', x.lower())

def webp(x):
    x2 = x.replace('.png', '') + '.webp'
    webplib.cwebp(x, x2, '-q 70')
    if removepng:
        os.system('rm -f ' + x)
    if sendimages2ftp:
        ftpput(x2)
    return x2


def asort(dict):
    #import collections
    if type(dict) == list:
        return dict  # allready
    #return collections.OrderedDict({k: v for k, v in sorted(dict.items(), key=lambda item: item[1])} )        
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1])}#tuples
    return sorted(dict.items(), key=operator.itemgetter(1), reverse=False)


def arsort(dict):
    #import collections
    if type(dict) == list:
        return dict  # allready        
    return {k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)} 
    return collections.OrderedDict({k: v for k, v in sorted(dict.items(), key=lambda item: item[1], reverse=True)} )        
    #return sorted(dict.items(), key=operator.itemgetter(1), reverse=True)    
    #

def nullValues(z):
    col_mask = z.isnull().any(axis=0)
    row_mask = z.isnull().any(axis=1)
    return z.loc[row_mask, col_mask]


def say(x):
    html("<script>say(\"" + str(x) + "\")</script>")


def html(x):
    import IPython
    IPython.display.display(IPython.display.HTML(x))
    return


def cleanData(inputDf, fillStrings='na', fillInt=0, considerEmpty=[
              np.inf, -np.inf, np.nan, 0, 'na', '']):
    #test = pd.read_csv('../input/test.csv')
    df = inputDf.copy(deep=True)
    rows = df.shape[0]
    cols = df.shape[1]
    dfs = df.size

    nanInfZeroNaEmpty = train.isin(
        considerEmpty).sum().sort_values(ascending=False)
    nv = nanInfZeroNaEmpty.sum()

    p('Total Rows in dataset : ' + str(rows))
    p('Total Cols in dataset : ' + str(cols))
    p('Total Cells : ' + str(dfs))
    p('Cells containing null,inf,NaN,0 or empty values : ' + str(nv) +
          ' ( ' + str(round(nv * 100 / dfs, 2)) + '% )')  # Diagnose null columns
    p('_' * 80)
    # Fournissent une bonne indication des colonnes à dropper pour le modèke
    for i in nanInfZeroNaEmpty.index:
        nbempty = nanInfZeroNaEmpty[i]
        per = round(nbempty * 100 / rows, 2)
        p(i + ' => empty : ' + str(nbempty) + ' ' + str(per) + '%')

    p('_' * 140)

    for i in df.columns:
        if df.dtypes[i] == object:
            df[i].fillna(fillStrings, inplace=True)  # as strings ! boljemoi !
        else:
            df.replace([np.inf, -np.inf, np.nan], fillInt,
                       inplace=True)  # replace infinite by Nan
    # df.fillna(df.mean(),inplace=True)#Numeric types corrected
    # df.fillna(fillInt,inplace=True)#2 rows of Nan -> is valid either for strings or numeric datatype
    # dtype as str please !!

    p('_' * 140)
    # CAUTION !!! WONT FIT INTO MODEL OTHERWISE
    p('Cells with null values then : ' + str(nullValues(df).size))
    p('_' * 140)
    return df

#gets the closest scope first
def rg(x,html=False):
  import requests
  if('.php' in x) | ('.html' in x):
    html=True
    r=requests.get(GG('webRepo')+GG('sftp')['cd']+'/'+x)
  else:
    r=requests.get(GG('webRepo')+GG('sftp')['cd']+'/'+x, stream=True)
  if(r.status_code==200):
    #p('rg:get: ',x);
    #print(r.text)#r.contents are b-encoded
    if html:
      FPC(x,r.text)#unicode
    else:
      createDirs(x)
      with open(x,'wb') as f:
        f.write(r.raw.read())  
    return True;
  return False;

def ftpget(fn, cd=0):
    if(type(fn) == str):  # solo
        fn = [fn]
    if (not GG('useFTP')) | (GG('sftp')['h']=='-'):
        oks=[]
        for i in fn:
            if(rg(i)):
                oks+=[i]
        p('rg get ok:' + ','.join(oks))
        return oks;
        p('ftp disabled')
    if(cd == 0):
        cd = GG('sftp')['cd']
        
    with pysftp.Connection(GG('sftp')['h'], username=GG('sftp')['u'], password=GG('sftp')['p'], cnopts=cnopts) as connection:
        with connection.cd(cd):
            for i in fn:
                connection.get(i)
            if GG('verbose'):
                p('get:' + ','.join(fn))


def ftpputzip(fn, cd=0):    
    if (not GG('useFTP')) | (GG('sftp')['h']=='-'):
        p('ftp disabled')
        return;
    if(type(fn) == str):  # solo
        fn = [fn]
    zipped = []
    for i in fn:
        os.system('rm -f ' + i + '.zip')
        os.system('zip ' + i + '.zip ' + i)
        zipped.append(i + '.zip')
    ftpput(zipped, cd)


def ftpexists(fn, cd=0):
    if(cd == 0):
        cd = GG('sftp')['cd']
    #liste = ftpls(cd)
    if(fn in GG('ftplist')):
        return True
    return False

def getFile(fns, sep='\t'):
    '''     get a local or distant file, either in a nested subdir, compressed or raw       '''
    notFound = []
    found = []
    if(type(fns) == str):  # solo
        fns = [fns]
    for fn in fns:
        #p(fn)
        fn2=fn
        
        if(fn.find('/')>1):#target has subdir
            fn2=fn.split('/')
            fn2=fn2[-1]#last one
            
        if(os.path.exists(fn2)):
            found.append(fn2)
            continue
        elif(os.path.exists(fn2 + '.zip')):
            o = subprocess.check_output('unzip -o ' + fn2 + '.zip', shell=True)
#isApple = True if fruit == 'Apple' else False
            if verbose:
                p(o)
            # os.system('unzip -o '+fn+'.zip')#unzip
            found.append(fn2)
            continue
        elif(os.path.exists(fn2 + '.tgz')):
            o = subprocess.check_output('tar xf ' + fn2 + '.tgz', shell=True)
#isApple = True if fruit == 'Apple' else False
            if verbose:
                p(o)
            # os.system('unzip -o '+fn+'.zip')#unzip
            found.append(fn2)
            continue
        elif(ftpexists(fn)):
            ftpget(fn)
            found.append(fn2)
            continue
        elif(ftpexists(fn + '.zip')):
            ftpget(fn + '.zip')
            o = subprocess.check_output('unzip -o ' + fn2 + '.zip', shell=True)
            if verbose:
                p(o)
            # os.system('unzip -o '+fn+'.zip')#unzip
            found.append(fn2)
            continue
        elif(ftpexists(fn + '.tgz')):
            ftpget(fn + '.tgz')
            o = subprocess.check_output('tar xf ' + fn2 + '.tgz', shell=True)
            if verbose:
                p(o)
            # os.system('unzip -o '+fn+'.zip')#unzip
            found.append(fn2)
            continue
        notFound.append(fn)
    if verbose:
        if(len(found)):
            p('Getfile:Found:' + ','.join(found))
        if(len(notFound)):
            p('Getfile:NotFound:' + ','.join(notFound))
    return len(found)

def nf(a):
    return;
    
#gv.a['ftplist']=[]
def fcb(a):
    gv.a['ftplist']+=[a.lstrip('./')]    
    
def ftpls(cd=0):
    if (not GG('useFTP')) | (GG('sftp')['h']=='-'):
        rg('list.php')
        x=fgc('list.php')
        SG('ftplist',x.split(','))
        return x.split(',')
    
    if(cd == 0):
        cd = GG('sftp')['cd']
    with pysftp.Connection(GG('sftp')['h'], username=GG('sftp')['u'], password=GG('sftp')['p'], cnopts=cnopts) as connection:
        with connection.cd(cd):
            gv.a['ftplist']=[]
            connection.walktree('.',fcb,nf,nf)#and recursive ?
            return gv.a['ftplist']
            #and recursive ?
            ftplist=connection.listdir()
            SG('ftplist',ftplist)
            #walktree            
    return ftplist
            
def createDirs(x):
    if('/' in x):
        fns=x.split('/')
        final=fns.pop();
        path='/'.join(fns)
        os.makedirs(path,exist_ok=True)
        
def FPC(fn, data):
    createDirs(fn)
    x = str(data).strip("\n\r ")
    #p(x,end='.')
    myfile = open(fn,'w',newline='\n')
    myfile.write(x)
    myfile.close()

def fgc(fn, join=True):
    lines = []
    with open(fn,'r') as f:
        lines += f.read().splitlines()
    if join:
        return "\n".join(lines)
    return lines

def nf(a=0,b=0,c=0,d=0,e=0,f=0,g=0,h=0,i=0,j=0,end=0):
    return;
    
#disable it : p=nf
def disablePrint():
    global p
    p=nf
    echo=nf
def enablePrint():
    global p
    p=print
    echo=print

def fgcj(fn):
    with open(fn + '.json') as json_file:
        return json.load(json_file)


def FPCJ(fn, data):  # r,b
    with open(fn + '.json', 'w') as json_file:
        json.dump(data, json_file)


def uniqueValuesPerColumn(sc, df2):
    return df2.groupby(sc)[sc].agg(
        ['count']).sort_values(
        by='count',
        ascending=False)

def lemitizeWords(text):
    words=token.tokenize(text)
    listLemma=[]
    for w in words:
        x=lemma.lemmatize(w, pos="v")
        listLemma.append(x)
    return ' '.join(map(str, listLemma))

def stopWordsRemove(text):
    words=token.tokenize(text)
    filtered = [w for w in words if not w in stop_words]
    return ' '.join(map(str, filtered))        
        
def extractionMots(x):
    import re
    import unidecode
    x=x.lower()
    x = unidecode.unidecode(x)
    notags=re.sub('<[^<]+?>', '', x)#suppression tags ouverture et fermeture ( conservant leur contenus interne : code, texte mis en forme, etc .. )
    noHTMLentities=re.sub('&[^;]{1,9};', '', notags)#cleanup &amp; &gt &lt
    stripped=re.sub(r"[^a-z0-9',.]+",' ', noHTMLentities)#autres que caractères de base
    lemitized=lemitizeWords(stripped)
    noStopWords=stopWordsRemove(lemitized)#i => retire du sens à la phrase conserve plupart mots clefs
    bouillie=re.sub(r"[',. ]+",' ',noStopWords).strip()#bouillie de mots
    #trimSingleLetters : I
    return trimAloneNumbers(bouillie)

def stripSimpleTags(x):
  import re
  x=re.sub('<','',x)#retrait de gauche
  x=re.sub('>',';',x)#remplacement par un séparateur plus commun
  x=re.sub(' +',' ',x)#multiples espaces potentiel par un unique
  return x.lower().strip('; ')#trim
  #return re.sub(' +',' ', re.sub('<|>',' ', x))
  #x='6 python pandas 34 scikit-learn a7s pmml 7'

def trimAloneNumbers(x):
  return re.sub('^[0-9]+\s|\s[0-9]+\s|\s[0-9]+$', '',x).strip()  

def trimSingleLetters(x):
  return re.sub('^[a-z]{1}\s|\s[a-z]{1}\s|\s[a-z]{1}$', '',x).strip()    

def arrayCount(x):
  return len(x)
  
def uniqueValues(x):
  x = np.array(x) 
  return list(np.unique(x)) 

def extractTagsn(x):
  return re.findall(r'(?<=\<)[^\<\>]+(?=\>)', x)
  
def extractTags(x,join=1):
  res=re.findall(r'(?<=\<)[^\<\>]+(?=\>)', x)
  if join:
    return ' '.join(res)
  return res
  

def ftpput(fn, cd=0, aszip=0):
    if (not GG('useFTP')) | (GG('sftp')['h']=='-'):
        p('ftp offline');
        return;
    
    if(type(fn) == str):  # solo
        fn = [fn]
    if(cd == 0):
        cd = GG('sftp')['cd']
        
    with pysftp.Connection(GG('sftp')['h'], username=GG('sftp')['u'], password=GG('sftp')['p'], cnopts=cnopts) as connection:
        with connection.cd(cd):
            for i in fn:
                if(GG('verbose')==True):
                    p(i.split('.')[-1], ' ',i, ' ' ,os.path.exists(i))
                if(not os.path.exists(i)):
                  p('!!!!',i,' not exists')
                  assert(False)
                  continue
                if((i.split('.')[-1] not in ['zip', 'jpg', 'png', 'webp', 'tgz', 'mp4', 'mp3', 'mkv','pik']) & (os.path.getsize(i) > 5000000 | aszip)):
                    os.system('rm -f ' + i + '.zip')
                    os.system('zip ' + i + '.zip ' + i)
                    i = i + '.zip'
                    
                connection.put(i)
                now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")  # -%s
                              
                if re.search("\.(jpg|png|webp)",i):
                    p('put : ' + GG('sftp')['web'] + GG('sftp')['cd'] + '/' + i + '?a=' + now)  # !
                    
                ftplist=GG('ftplist');ftplist+=[i];SG('ftplist',ftplist)#add to exported files list


def pltinit(df, i, j, title=False,width=6,height=6):    
    fig, ax = plt.subplots(figsize=(height,width))
    fig.patch.set_facecolor('white')
    x = df[i]
    y = df[j]
    if(title==False):
        title=i + ' vs ' + j
    plt.title(title)
    plt.xlabel(i)
    plt.ylabel(j)
    return [x, y, filename(i + '.' + j), fig, ax]

def plot(df, i='x', j='y', rotate=False, fn=False, title=False,width=6,height=6):
    '''plot(var)'''
    global x    
    if type(df)==list:    
        x={}
        k=0
        for val in df:
            x[k]=val
            k+=1
        df=x

    if type(df)==dict:    
        df=pd.DataFrame.from_dict({i:list(df.keys()),j:list(df.values())}).sort_values(by=i,ascending=True)

    x2, y, fn2, fig, ax = pltinit(df, i, j, title,height,width)
    if(fn==False):
        fn=fn2
    # bestCorrelationsKeys.keys():
    plt.plot(x2, y)
    if rotate:
        plt.xticks(rotation=rotate)
    fn = 'plot' + fn + '.png'
    plt.savefig(fn, bbox_inches='tight')
    webp(fn)
    show()


def scatter1(df=False, i=False, j=False, ts=0,x=False,y=False,color=False,fn='',cmap='brg',o=1):
    if((isinstance(x,np.ndarray)) & (isinstance(y,np.ndarray))):
        pass;
    else: 
        x, y, fn, fig, ax = pltinit(df, i, j)
    if ts:
        plt.gca().set_yticklabels(
            backgroundcolor='white',
            labels=y,
            rotation=(0),
            fontsize=ts,
            linespacing=ts)
        # ax.tick_params(axis='y',which='major',pad=ts)
        #ax.set_yticklabels(labels=y,rotation = (45), fontsize = 10, va='bottom', ha='left')
    if(isinstance(color,pd.Series)):
        #plt.scatter(x, y,c=color,cmap=cmap,alpha=o)
        sns.scatterplot(x, y, alpha=o, hue=color, palette=sns.color_palette(cmap,unik(color).shape[0]))
    else:
        plt.scatter(x, y)
    plt.title(fn)
    fn='scatter' + fn + '.png';
    plt.savefig(fn, bbox_inches='tight')
    webp(fn)
    show()

# type(sup0)==pandas.core.series.Series


def scatter(
    df,
    a='',
    b='',
    ts=0,
    minx=0,
    maxx=0,
    miny=0,
    maxy=0,
    opacity=0.02,
    color='blue',
    size=2,
    axis=0,
    xscale=0,
    yscale=0,
    reg=0,
        fn=0):
    if(fn == 0):
        fn = filename(a + '.' + b + '.png')
    # recombiné avec son index
    if(type(df) == pd.core.series.Series):  # {
        if(len(df) == 0):
            p('empty.. skipping')
            return 0
        # x=df.index
        # p('serie')
        if(maxy == 0):  # maxx or de propos
            maxy = df.max()
        x = range(df.shape[0])

        if(axis):
            axis.scatter(x, df, s=size, alpha=opacity)
            # axis.set_title(fn)
            #  ax.set_xlim(minx,maxx);ax.set_ylim(miny,maxy)
            axis.set_xlabel(a)
            axis.set_ylabel(b)
            axis.set_facecolor('white')
            if(yscale):
                axis.set_yscale(yscale)
            if(xscale):
                axis.set_xscale(xscale)
            return 1

        plt.figure().set_facecolor('white')
        plt.scatter(x, df, s=size, alpha=opacity)
        # plt.title(fn)
        if maxy:
            plt.ylim(miny, maxy)
        plt.xlabel(a)
        plt.ylabel(b)
        if(yscale):
            plt.yscale(yscale)
        if(xscale):
            plt.xscale(xscale)
        plt.savefig(fn, bbox_inches='tight')
        # ftpput(fn)
        webp(fn)
        show()
        return fn
# }end pandas serie
# Sinon Dataframe ordinaire
    elif(axis == 0):
        # p('joinplot')
        x = df[a]
        y = df[b]
        args = {
            'x': x,
            'y': y,
            'data': df,
            'joint_kws': {
                'alpha': opacity,
                'color': color}}
        if(reg):
            args['kind'] = 'reg'
            args['joint_kws'] = {'color': color}
            args['scatter_kws'] = {'alpha': opacity}

        # p(args);#TypeError: regplot() got an unexpected keyword argument
        # 'alpha'
        sns.jointplot(**args).savefig(fn, bbox_inches='tight')
        webp(fn)
        show()
        return fn
# Rendu sur un axe ..
    x, y, fn2, fig, ax = pltinit(df, a, b, axis)

    if(maxx == 0):
        maxx = x.max()
    if(maxy != 0):
        maxy = y.max()

    if(axis != 0):  # specifié non généré
        axis.set_xlabel(a)
        axis.set_ylabel(b)
        axis.scatter(x, y, alpha=opacity, c=color, s=size)
        ax = axis

    ax.set_facecolor('white')

    if(xscale):
        ax.set_xscale(xscale)
    if(yscale):
        ax.set_yscale(yscale)
    return 1

    if(ts & False):
        plt.gca().set_yticklabels(
            backgroundcolor='white',
            labels=y,
            rotation=(0),
            fontsize=ts,
            linespacing=ts)
        # ax.tick_params(axis='y',which='major',pad=ts)
        #ax.set_yticklabels(labels=y,rotation = (45), fontsize = 10, va='bottom', ha='left')
    if(axis == 0):
        if False:
            plt.scatter(x, y, alpha=opacity, c=color, s=size)
            show()
            plt.savefig(fn, bbox_inches='tight')
            webp(fn)


def md5(x):
    return hashlib.md5(bencode.bencode(x)).hexdigest()


def NewModel1(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(y_size, activation='linear'))
    return(t_model)


def NewModel2(x_size, y_size):
    t_model = Sequential()
    t_model.add(Dense(100, activation="tanh", input_shape=(x_size,)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(50, activation="relu"))
    t_model.add(Dense(20, activation="relu"))
    t_model.add(Dense(y_size, activation='linear'))
    return(t_model)


def NewModel3(x_size, y_size):
    t_model = Sequential()
    t_model.add(
        Dense(
            80,
            activation="tanh",
            kernel_initializer='normal',
            input_shape=(
                x_size,
            )))
    t_model.add(Dropout(0.2))
    t_model.add(
        Dense(
            120,
            activation="relu",
            kernel_initializer='normal',
            kernel_regularizer=regularizers.l1(0.01),
            bias_regularizer=regularizers.l1(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(
        Dense(
            20,
            activation="relu",
            kernel_initializer='normal',
            kernel_regularizer=regularizers.l1_l2(0.01),
            bias_regularizer=regularizers.l1_l2(0.01)))
    t_model.add(Dropout(0.1))
    t_model.add(Dense(10, activation="relu", kernel_initializer='normal'))
    t_model.add(Dropout(0.0))
    t_model.add(Dense(y_size, activation='linear'))
    return(t_model)


def nn4(x_size, y_size):
    model = Sequential([
        Dense(32, activation='relu', input_shape=(x_size,)),
        Dense(32, activation='relu'),
        Dense(y_size, activation='sigmoid')
    ])
    # model.compile(optimizer='sgd',
    return model


def histogram(x, name, score):
    # k=x.history.keys();p(k)#dict_keys(['val_loss', 'val_mean_absolute_error', 'loss', 'mean_absolute_error'])#"mean_squared_error"
    # mean_squared_logarithmic_error
    first = x.history['mean_squared_error'][:1]
    # TypeError: list indices must be integers or slices, not tuple
    last = x.history['mean_squared_error'][-1:]
    # p(first[0]);p(last[0])
    progress = first[0] - last[0]
    p('Progress:' + str(progress))
    # plusieurs champs k
    for i in x.history.keys():
        plt.plot(x.history[i], label=i)
    plt.title(str(score) + '/' + name)
    x = filename(str(name)) + '.histogram.png'
    plt.savefig(x, bbox_inches='tight')
    ftpput(x)
    plt.close()
    return progress
    show()

# Fit and Predict at the same time !!!


def fit(x, fn=0, noload=0):
    global x1, x2, y1, y2, ep, bs, k, pred, acy, k, toGuess
    res = 0
    history = 0
    if(fn):  # changmenet de shape de données en input ..
        if(noload == 0 & os.path.isfile(fn)):
            getFile(fn)
            x = keras.models.load_model(fn)
            p('load model:' + fn)
            res = 1

        if(res == 0):
            p('generate model:' + fn)
            history = x.fit(
                x1,
                x2,
                validation_data=(
                    y1,
                    y2),
                epochs=ep,
                batch_size=bs,
                shuffle=True,
                verbose=0)
            p('_' * 160)
            # history=x.fit(standardize(x1),standardize(x2),validation_data=(standardize(y1),standardize(y2)),epochs=ep,batch_size=bs,shuffle=True,verbose=2)
            x.save(fn)
            ftpput(fn)  # zipped ?

    nfm[k] = x
    pred[toGuess][k] = x.predict(y1)  # mean_squared_log_error
#!!!:ValueError: Mean Squared Logarithmic Error cannot be used when targets contain negative values.
    acy[toGuess][k] = round(mean_squared_error(
        y2, pred[toGuess][k]) ** (1 / 2))
    say(acy[toGuess][k])
    p(k + ' => ' + str(acy[toGuess][k]))
    if(history):
        histogram(history, k, acy[toGuess][k])
    p('_' * 120)

# celles du module ?


def globkeys():
    p(globals().keys())


def r2(a, b):
    return stats.pearsonr(a, b)[0] ** 2


def standardize(df):
    mean = np.mean(df, axis=0)
    std = np.std(df, axis=0)  # +0.000001
    return (df - mean) / std


def FPCP(fn, data):
    os.system('rm -f ' + fn + '.pickle ' + fn + '.pickle.zip')
    f = open(fn + '.pickle', 'wb')
    pickle.dump(data, f, protocol=-1)
# d=pickle.dumps(data);f.write(d);f.close();
# can't pickle _thread.RLock objects
    
#ModuleNotFoundError: No module named 'sklearn.linear_model._logistic
SG('noClassRewrite',True)
class rewriteClass(pickle.Unpickler):
    def find_class(self, module, name):
        if GG('noClassRewrite'):#origin
            return super(rewriteClass, self).find_class(module, name)
        renamed_module = module
        a=module.split('.')        
        if((a[-1][0]=='_') & (a[-1] not in ['_pickle'])):
            p('rewrited',a[-1],a[-1][0])
            del(a[-1])
            renamed_module='.'.join(a)
        
        if module == "sklearn.linear_model._logistic":
            renamed_module = "sklearn.linear_model"
        return super(rewriteClass, self).find_class(renamed_module, name)


def fgcp(fn):
    fn = fn + '.pickle'
    getFile(fn)
    if verbose:
        p('size:', fn, ' : ', os.path.getsize(fn) / 1024 / 1024)
    if(os.path.isfile(fn)):
        gc.disable()
        data = open(fn, "rb")
        ret = rewriteClass(data).load()
        #ret = pickle.load(data)
        data.close()
        gc.enable()
        return ret
    return 0

def loadIfNotSet(x):
    if x not in globals().keys():
        load(x)
    
def mail(x):
    import requests
    url = 'https://1.x24.fr/a/bus.php'    
    r = requests.post(url, data={'mail': x})
    
def message(x):
    import requests
    url = 'https://1.x24.fr/a/bus.php'    
    r = requests.post(url, data={0: x})
    r = requests.post(url, data={'mail': x})


# snscat
def snsscat(
    df,
    a='',
    b='',
    ts=0,
    minx=0,
    maxx=False,
    miny=0,
    maxy=False,
    opacity=0.02,
    color='blue',
    size=2,
    axis=0,
    xscale=0,
    yscale=0,
    fn=False,
    kind='scatter',
        height=24):
    if(fn == False):
        fn = filename(a + '.' + b + '.png')
    x = df[a]
    y = df[b]  # fig,ax=plt.subplots(1);#ax=ax.flatten()
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    corr = stats.pearsonr(x, y)[0] ** 2
    # x,y,fn2,fig,ax=pltinit(df,a,b,axis)
    # ax.scatter(x,y,alpha=opacity,c=color,s=size);
    if(maxx == False):
        maxx = x.max()
    if(maxy == False):
        maxy = y.max()
# https://stackoverflow.com/questions/36191906/rescale-axis-on-seaborn-jointgrid-kde-marginal-plots    'ax':ax, 'height':height,
# https://seaborn.pydata.org/generated/seaborn.JointGrid.html
    args = {
        'height': height,
        'dropna': True,
        'kind': kind,
        'x': x,
        'y': y,
        'data': df,
        'joint_kws': {
            'alpha': opacity,
            'color': color},
        'xlim': [
            minx,
            maxx],
        'ylim': [
            miny,
            maxy],
        'color': 'red'}
    args['stat_func'] = r2
    args['marginal_kws'] = dict(bins=15, rug=True)
    args['annot_kws'] = dict(stat="r")  # joinplot annot_kws
# N'apparait pas ..

    if(kind in ['reg']):
        args['joint_kws'] = {'color': color}  # reg no opacity
        # not hex,nor kde
        args['line_kws'] = {
            'label': "y={0:.1f}x+{1:.1f}".format(slope, intercept)}
        args['scatter_kws'] = {'alpha': opacity}  # ,nor kde
    if(kind in ['kde']):
        args['joint_kws'] = {'color': color}
        args['marginal_kws'] = {}  # no bins property
        pass

    # p(args);#TypeError: regplot() got an unexpected keyword argument
    # 'alpha',dir(ax)
    g = sns.jointplot(**args)  # unpack
    ax = g.ax_joint
    # g=sns.JointGrid(**args)
    """
  g = sns.JointGrid(x="total_bill", y="tip", data=tips)
  g = g.plot_joint(plt.scatter,color="g", s=40, edgecolor="white")
  g = g.plot_marginals(sns.distplot, kde=False, color="g")
  g = g.annotate(stats.pearsonr)

  g = g.plot_joint(sns.kdeplot, cmap="Blues_d")
  g = g.plot_marginals(sns.kdeplot, shade=True)
  """
    if(xscale):
        ax.set_xscale('log')
        if(kind in ['reg', 'kde']):
            g.ax_marg_x.set_xscale('log')

    if(yscale):
        ax.set_yscale(yscale)
        if(kind in ['reg', 'kde']):
            g.ax_marg_y.set_yscale('log')
    # ax.set_xlabel(a);ax.set_ylabel(b);#déjà attribués
# AttributeError: 'JointGrid' object has no attribute 'set_xlim'
    if((kind in ['reg', 'kde']) & False):
        regline = g.ax_joint.get_lines()[0]
        regline.set_color('black')
        regline.set_zorder('99')
    #fig.text(1, 1, "An annotation", horizontalalignment='left', size='medium', color='black', weight='semibold')
    # dir(g)
    # rsquare = lambda a, b:
    #rsquare = lambda a, b: 2;g.annotate(rsquare)
    #ax = g.axes[0,0]
    plt.title(str(int(slope)) + '-' + str(int(intercept)))
    g.savefig(fn, bbox_inches='tight')
    webp(fn)
    # ftpput(fn)
    show()
    # plt.show()
    return 1


regplot = snsscat


def saveAsOne(
        _globals,
        exclusions=[],
        fn='allVars',
        include=False,
        backup=False):
    exclusions += 'Out,In,models,allVars,sftp'.split(',')
    _vars = {}
    _gk = list(_globals.keys())
    if(include):
        _gk = include
    for i in _gk:
        if i.startswith('_'):
            continue
        if i in _globals.keys():
            if type(
                    _globals[i]) in [
                    str,
                    dict,
                    list,
                    int,
                    pd.DataFrame,
                    pd.Series]:  # yyuuuuuu !
                _vars[i] = _globals[i]
    size = {}
    # x=%who_ls str dict list int DataFrame Series
    a = tim()
    for i in exclusions:
        if i in _vars.keys():
            del _vars[i]

    for i in _vars:  # compact
        size[i] = sys.getsizeof(_vars[i])
    p(arsort(size))
    #p(','.join(x)+':: saved')
    os.system('rm -f ' + fn + '.pickle ' + fn + '.pickle.zip')
    FPCP(fn, _vars)
    os.system('stat ' + fn + '.pickle')
    if(GG('verbose')==True):
        p('fs : ' + str(round(os.stat(fn + '.pickle').st_size / 1024 / 1024)),end='.') # + ' //--- saved in:' + str(round(tim() - a)) + 's'
    o = subprocess.check_output('zip ' +fn +'.pickle.zip ' +fn +'.pickle',shell=True)
    p(o)
    ftp = [fn + '.pickle.zip']

    if backup:
        now = datetime.datetime.now()
        fn2 = fn + '.' + now.strftime("%Y%m%d-%H%M") + '-backup.pickle.zip'
        os.system('cp ' + fn + '.pickle.zip ' + fn2)
        ftp += [fn2]

    ftpput(ftp)

def reduceList(n, iterable):
  from itertools import islice
  "Return first n items of the iterable as a list"
  return list(islice(iterable, n))    

take=reduceList

# globals().update(resume('allVars'))
def ramUsage():
    import psutil
    process = psutil.Process(os.getpid())
    return round(process.memory_info().rss/1024/1024,2)  # in M
    
#topramusageperkey
def ramUsagePerKey():
  gk=list(globals().keys())
  vpk={}
  for i in gk:
    if callable(globals()[i]):
      continue
    vpk[i]=sys.getsizeof(globals()[i])
  display(arsort(vpk))

def plot_feature_importances(feature_importances, title, feature_names, fn=0):
    # Normalize the importance values
    plt.rcParams["figure.figsize"] = (24, 6)
    feature_importances = 100.0 * \
        (feature_importances / max(feature_importances))
    # Sort the index values and flip them so that they arearranged in
    # decreasing order of importance
    index_sorted = np.flipud(np.argsort(feature_importances))
    # p(index_sorted)
    # Center the location of the labels on the X-axis (for displaypurposes
    # only)
    pos = np.arange(index_sorted.shape[0]) + 0.5
    # Plot the bar graph
    plt.figure()
    plt.bar(pos, feature_importances[index_sorted], align='center')
    plt.xticks(pos, feature_names[index_sorted], rotation=90)
    plt.xlabel('Relative Importance')
    plt.title(title)

    if(fn == 0):
        fn = 'plotFeatImportance-' + title + '.png'
    plt.savefig(fn, bbox_inches='tight')
    webp(fn)
    show()


def non_shuffling_train_test_split(X, y, test_size=0.2):
    i = int((1 - test_size) * X.shape[0]) + 1
    X_train, X_test = np.split(X, [i])
    y_train, y_test = np.split(y, [i])
    return X_train, X_test, y_train, y_test


#x1, y1, x2, y2 = sklearn.model_selection.train_test_split(df2,results,train_size=0.8, test_size=0.2, random_state=100 )
shuffleData = 0


def ShuffleOrNot(_df, results, train_size=0.8, test_size=0.2):
    if shuffleData:
        # TypeError: train_test_split train_size Singleton array array cannot
        # be considered a valid collection
        return sklearn.model_selection.train_test_split(
            _df,
            results,
            train_size=train_size,
            test_size=test_size)  # ,random_state=random_state
    p('non shuffled')
    return sklearn.model_selection.train_test_split(
        _df,
        results,
        train_size=train_size,
        test_size=test_size,
        shuffle=False)  # , stratify = None
    return non_shuffling_train_test_split(_df, results, test_size=test_size)


def loadData(f, sep=','):
    return pd.read_csv(f, sep=',')


def firstValueOf(x):
    v = x.value_counts()
    return v.index[0]

def vc(x):
  if type(x)==list:
    return unik(x);
  if type(x)==np.ndarray:
    from collections import Counter
    r = dict(Counter(x))
    return r
  p(type(x))
  v=x.value_counts()#print(v.sum())
  return v.sum()
    
def distinct(x):
    v = x.value_counts()
    # p(v.index[0])
    return ','.join(v.index)


def unikValuesPerDataframe(train):
    unikStringValuesPerColumn = {}
    # colonnesStrings=train.select_dtypes(include=['object']).columns
    colonnesStrings = train.columns.values
    for i in colonnesStrings:
        if i.startswith('z_'):
            continue
        suffix = ''
        if train.dtypes[i] == object:
            # cast All Strings to lowercase AND TRIM ?
            train[i] = train[i].str.lower()
        else:
            suffix = ' (mean:' + str(train[i].mean()) + ')'

        unikStringValuesPerColumn[i] = len(train[i].unique())
        p('_' * 120)
        p(' 🗲 Different values in ' + i + ' : ' +
              str(unikStringValuesPerColumn[i]) + suffix)
        p(train[train[i] != 0].groupby(i)[
              i].count().sort_values(ascending=False).head(5))

# _df[:1][columns]
# radar stuff


def radar(_df, title='radar', seuilSuppression=1):
    if type(_df) == pd.core.series.Series:
        _df = pd.DataFrame(_df).T

    fn = title + '.png'
    sel = _df.copy()
    # p(sel);p(sel['product_name'].values[0]);p(sel['code'].values[0])
    for i in sel.keys():
        # p(i);p(sel[i].values[0])
        if type(sel[i]) == 'object':
            del(sel[i])
        if(i in 'code,product_name'.split(',')):
            del(sel[i])
        elif(sel[i].values[0] < seuilSuppression):
            del(sel[i])
            # p('del:'+i)
# Exception: Data must be 1-dimensional
    radardf = pd.DataFrame(dict(r=sel.values[0], theta=sel.columns))
    fig = px.line_polar(
        radardf,
        r='r',
        theta='theta',
        line_close=True,
        title=title)
    fig.update_traces(fill='toself')
    return 1  # px never exports
    # fig.savefig(fn,bbox_inches='tight');webp(fn);fig.close()


def radar2(df, fn='radar'):
    plt.title(fn)
    fn += '.png'
    if type(df) == pd.core.series.Series:
        df = pd.DataFrame(df).T

    indexes = list(df.index.values)
    categories = list(df.columns.values)  # [1:]
    N = len(categories)
    # What will be the angle of each axis in the plot? (we divide the plot /
    # number of variable)
    angles = [n / float(N) * 2 * 3.14 for n in range(N)]
    #angles += angles[:1]
    # Initialise the spider plot
    ax = plt.subplot(111, polar=True)
    # We are going to plot the first line of the data frame.
    # But we need to repeat the first value to close the circular graph:
# TypeError: cannot do label indexing on <class
# 'pandas.core.indexes.category.CategoricalIndex'> with these indexers [0]
# of <class 'int'>
    values = df.loc[indexes[0]].values.flatten().tolist()
    #values += values[:1]
    # p(len(categories),len(angles),len(values),categories,angles,values)

    # Draw one axe per variable + add labels labels yet
    plt.xticks(angles, categories, color='grey', size=8)

    # Draw ylabels
    ax.set_rlabel_position(0)
    #plt.yticks([10,20,30], ["10","20","30"], color="grey", size=7);plt.ylim(0,40)
    ax.plot(angles, values, linewidth=1, linestyle='solid')
    ax.fill(angles, values, 'b', alpha=0.1)
    ax.figure.savefig(fn, bbox_inches='tight')
    webp(fn)
    plt.close()


def unik(x, aslist=False):
    if(type(x) == list):
        from collections import Counter
        r = dict(Counter(x))
        if aslist:
            return list(r.keys())
        return r
    if type(x) == np.ndarray:
        return set(x)
# if dataframe
    if aslist:
        return list(x.value_counts().keys())
    return x.value_counts()
    
unique=unik

def _save(
        _globals,
        exclusions=[],
        fn='allVars',
        include=False,
        backup=False,
        ftp=True,
        cleanup=False,
        zip=True,
        authTypes=[str,dict,list,int,np.ndarray,pd.DataFrame,pd.Series]
        ):
    
    ftplist=GG('ftplist')
    exclusions += 'Out,In,models,allVars,sftp'.split(',')
    size = {}
    dumpedsize = {}
    _vars = []
    _gk = list(_globals.keys())
    if(include):
        _gk = include
    for i in _gk:
        if i.startswith('_'):  # exclude all variables starting with underscores ( portée locale uniquement )
            continue
        if i in _globals.keys():
            if callable(_globals[i]):
                continue
            if(authTypes):
                if type(_globals[i]) in authTypes: 
                    _vars += [i]
            else:
                _vars += [i]

    # x=%who_ls str dict list int DataFrame Series
    a = tim()
    for i in _vars:  # compact
        if i in exclusions:
            continue  # skipit
        FPCP(i, _globals[i])
        ftplist+=[i]
        size[i] = sys.getsizeof(_globals[i])
        dumpedsize[i] = round(os.stat(i + '.pickle').st_size)  # /1024/1024
    # p(arsort(size))
    #return dumpedsize
    #SG('ds',dumpedsize)
    sumOfFiles=sum(dumpedsize.values())
    #p('sumOfFiles',sumOfFiles)

    if((zip) & (sumOfFiles > 2000000)):
        #p(fn)
        os.system('rm -f ' + fn + '.tgz')
        fl = []
        for i in size.keys():
            fl += [i + '.pickle']
        #p(fl)
        FPC('filelist.list', '\n'.join(fl))
        #os.system('pf '+fl+'>filelist.list')
        os.system('tar czf ' + fn + '.tgz --files-from=filelist.list')
        if(GG('verbose')==True):
            p('fs : ' + str(round(os.stat(fn + '.tgz').st_size / 1024 / 1024)) + ' //--- saved in:' + str(round(tim() - a)) + 's')
        fn+='.tgz'

        if(cleanup):
            for i in size.keys():
                os.system('rm -f ' + i + '.pickle')
    elif(len(size)==0):
        p('nothing to save',_gk);
        return;
    else:#not zip, is single text value :)
        fn=list(size.keys())[0]+'.pickle'
        
    if(ftp):    
        ftp = [fn]
        if backup:
            now = datetime.datetime.now()
            fn2 = fn + '.' + now.strftime("%Y%m%d-%H%M") + '.back'
            os.system('cp ' + fn +' ' + fn2)
            ftp += [fn2]
            
        ftpput(ftp)
    SG('ftplist',ftplist)

def resume(fn='allVars'):  # restore
    import tensorflow
    a = tim()
    _allVars = {}
    found=getFile([fn + '.pickle',fn + '.tgz',fn + '.h5',fn + '.joblib'])
    if GG('verbose'):
        p('Resuming : ',fn,',exists:',os.path.exists(fn + '.pickle'),' or tgz:',os.path.exists(fn + '.tgz'))        
        p('files found:', found)#getFile checks for zip and unzips them ,fn + '.h5.zip',fn + '.joblib.zip', fn + '.zip'

    if(os.path.exists(fn + '.pickle')):
        _allVars[fn] = fgcp(fn)  # nom de variable individuelle
    elif(os.path.exists(fn + '.joblib')):
        _allVars[fn] = joblib.load(fn + '.joblib')
    elif(os.path.exists(fn + '.h5')):
        _allVars[fn] = tensorflow.keras.models.load_model(fn + '.h5')
    elif(os.path.exists(fn + '.tgz')):
        #subprocess.check_output('tar -ztf allVars.tgz')
        os.system('tar -ztf ' + fn + '.tgz > filelist.list')
        o = fgc('filelist.list', join=False)
        if GG('verbose'):
            p('list of files within tgz:', o)
        os.system('tar xf ' + fn + '.tgz')
        #o=subprocess.check_output('tar xf '+fn+'.tgz');p(o);
        for i in o:
            i2 = i.replace('.pickle', '')
            _allVars[i2] = fgcp(i2)
    return _allVars

#load = restore = resume


def uniqueList(list1):
    x = np.array(list1)
    return list(np.unique(x))

def countwords(phrase):
    return len(re.findall(r'\w+', phrase)) 

def kde(var,title=0, fn=0, quantilex=1, ax=False):
    ''' get kde for variable '''
    if type(var)==list:#serie of dataframe
        var=pd.DataFrame(var)[0]
        
    save = 0
    if not ax:
        save = 1
        f, ax = plt.subplots(1, 1)
        f.patch.set_facecolor('white')
    if (title==0) & (fn!=0):
        title=fn
    ax.set_title(title)
    # Avec les limites adaptées à chaque type de données
    ax.set_xlim(0, var.quantile(quantilex))
    var.plot(kind='kde', ax=ax).yscale = 'log'
    if (fn!=0) & (save==1):
        f.savefig(fn, bbox_inches='tight')
        webp(fn)
        plt.close()

def MagicPca(df, clusters, fn, log=False):
    _x = df.copy()
    cols = list(_x.columns.values)
    p(cols)
    if log:
        for i in cols:
            _x[i] = _x[i].apply(np.log).round(3)
        # passage au log ..
        _x.replace([np.inf, -np.inf, np.nan], 0, inplace=True)

    _std_scale = preprocessing.StandardScaler().fit(_x)
    _x_scaled = _std_scale.transform(_x)  # matrice autres paramètres
    # réduire à deux dimensions, dont les vecteurs ci dessous représentent les
    # contributions
    pca = decomposition.PCA(n_components=2, random_state=42)
    pca.fit(_x_scaled)
    pcs = pca.components_

    X_projected = pca.transform(_x_scaled)
    return pcaScatter(X_projected, pcs, clusters, cols, fn)


def pcaScatter(
        X_projected,
        clusters,
        cols,
        fn,
        alpha=1,
        pcs=False,
        centers=False,
        scaleEigenVectors=False,
        title=False,
        displayTriples=False,
        displayPairs=True,
        xmin=0,
        xmax=0,
        ymin=0,
        ymax=0):
  # p(type(centers),type(centers)==np.ndarray)
  xminFixed=0
  if(xmin!=0):
    xminFixed=1
  nbDimensions=X_projected.shape[1]
  #nbColonnes,vecteurs
  pairs=list(itertools.combinations(range(0,nbDimensions),2))
  #et les triples ?
  
  dims = {}
  k = 0

  if((nbDimensions>2) & displayTriples):
    triples=list(itertools.combinations(range(0,nbDimensions),3))
    for triple in triples:
      p(triple)
      xs=pca_projected[:,triple[0]];ys=pca_projected[:,triple[1]];zs=pca_projected[:,triple[2]]
      px.scatter_3d(pd.DataFrame.from_dict({'Dim'+str(triple[0]):xs,'Dim'+str(triple[1]):ys,'Dim'+str(triple[2]):zs,'color':clusters}),
                    x='Dim'+str(triple[0]),y='Dim'+str(triple[1]),z='Dim'+str(triple[2]),
                    color='color',color_continuous_scale=rgb,opacity=.3).show()

  if(displayPairs):
    for pair in pairs:
      p('dimensions',pair)
      xs = X_projected[:, pair[0]]
      ys = X_projected[:, pair[1]]
      if(xminFixed==0):
        xmax = xs.max()
        xmin = xs.min()
        ymax = ys.max()
        ymin = ys.min()

      xscale = (xmax - xmin) / 2
      yscale = (ymax - ymin) / 2
      mlim = min([abs(ymin), abs(ymax), abs(xmin), abs(xmax)])

      sns.scatterplot(xs, ys, alpha=alpha, hue=clusters, palette=sns.color_palette(
        'brg', np.unique(clusters).shape[0]))  
      
      if type(pcs) == np.ndarray:    
        nbColumns = pcs.shape[1]                      
        for i in list(range(nbColumns)):
            x1 = pcs[pair[0], i] * mlim  # xscale*0.5; exagérées
            y1 = pcs[pair[1], i] * mlim  # yscale*0.5;#deux dimensions

            if scaleEigenVectors:
                x1 = pcs[pair[0], i] * xscale * 0.5  # ; exagérées
                y1 = pcs[pair[1], i] * xscale * 0.5

            plt.arrow(
                0,
                0,
                x1 * 0.8,
                y1 * 0.8,
                width=0.04,
                color='k',
                alpha=0.8)
            plt.text(
                x1 * 0.9,
                y1 * 0.9,
                cols[i],
                color='k',
                ha='center',
                va='center',
                fontsize=32)
            if(xminFixed==0):
              if(x1 > xmax):
                  xmax = x1
              elif(x1 < xmin):
                  xmin = x1
              if(y1 > ymax):
                  ymax = y1
              elif(y1 < ymin):
                  ymin = y1
                  
      plt.xlabel('dim:'+str(pair[0]))
      plt.ylabel('dim:'+str(pair[1]))
      plt.xlim(xmin, xmax)
      plt.ylim(ymin, ymax)

      if(type(centers) == np.ndarray):
        plt.scatter(centers[:, pair[0]], centers[:, pair[1]],marker='x', color='k', linewidths=2, s=160)

      if title:
        plt.title('dim'+str(pair[0])+'-'+str(pair[1])+'-'+title)

      fn2='nbdim-'+str(nbDimensions)+'-dim'+str(pair[0])+'-'+str(pair[1])+'-'+fn
      plt.savefig(fn2, bbox_inches='tight')
      webp(fn2)
      plt.close()

  if type(pcs) == np.ndarray:
      for i in pcs:
          l = 0
          dims[k] = {}
          for j in i:
              dims[k][cols[l]] = j
              l += 1
          k += 1

  return dims

def splitOnSpace(x):
  return x.split(' ')
#lst1 not in lst2
def diff(lst1, lst2):
    lst3 = [value for value in lst1 if value not in lst2]
    return lst3

#lst1 in lst2    
def intersection(lst1, lst2):
    return list(set(lst1).intersection(set(lst2)))
    #
    lst3 = [value for value in lst1 if value in lst2]
    return lst3
    
intersect = intersection


def md5(x):
    res = hashlib.md5(str(x).encode())
    return res.hexdigest()

def unikStrValues(x):
  return ','.join(sorted(pd.Series.unique(x)))
    
def labelsToNumeric(x):
    labelencoder = LabelEncoder()
    return labelencoder.fit_transform(x),labelencoder    
    
def toLogScaled(odf, cols):
    labenc=0
    df=odf.copy()
    if 'category' in list(df.columns.values):
        df['category'],labenc=labelsToNumeric(df['category'])
        
    for i in cols:
        df[i] = df[i].apply(np.log).round(3)    

    # passage au log .. pas d'erreur de conversion float -inf .???
    df.replace([np.inf, -np.inf, np.nan], 0, inplace=True)
    _scaler = StandardScaler()
    # Input contains infinity or a value too large for dtype('float64').
    df_scaled = _scaler.fit_transform(df[cols])
    df_scaled = pd.DataFrame(df_scaled, index=df.index, columns=cols)
    return df_scaled,labenc
    
def percentageByCat(x,nb=10):
    global _x
    tot=len(x)
    ret={}
    _x=x.value_counts()[:nb]
    for i in _x.index:
        ret[i]=round(_x[i]*100/tot,2)
    return json.dumps(ret)
    
def cleanup():
    os.system('rm -f *.pickle')# *.tgz
    
def delm(x):
  print('deleting:'+x)
  os.system('rm -f '+x+'.pickle '+x+'.tgz')

def deli(xs,callback=nf):
    p(ramUsage(),end=':')
    for x in xs.split(','):
        if x in globals().keys():
            globals()[x]=None
            del(globals()[x])
            callback(x);
            p(x,end=',')   
            gc.collect()
    p('=>',ramUsage())

def plot_confusion_matrix(cm,target_names,title='Confusion matrix',cmap=None,normalize=True,w=18,h=18):
    #http://scikit-learn.org/stable/auto_examples/model_selection/plot_confusion_matrix.html
    import matplotlib.pyplot as plt
    import numpy as np
    import itertools
    accuracy = np.trace(cm) / np.sum(cm).astype('float')
    misclass = 1 - accuracy

    if cmap is None:
        cmap = plt.get_cmap('Blues')

    plt.figure(figsize=(w, h))
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()

    if target_names is not None:
        tick_marks = np.arange(len(target_names))
        plt.xticks(tick_marks, target_names, rotation=45)
        plt.yticks(tick_marks, target_names)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]


    thresh = cm.max() / 1.5 if normalize else cm.max() / 2
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        if normalize:
            plt.text(j, i, "{:0.4f}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")
        else:
            plt.text(j, i, "{:,}".format(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black")


    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label\naccuracy={:0.4f}; misclass={:0.4f}'.format(accuracy, misclass))
    plt.savefig(title+'.png', bbox_inches='tight');webp(title+'.png');plt.close();

if ' heartbeats & monitoring ':
  threads=[]
  import threading,time,requests
  class backgroundThread: 
    def __init__(self): 
      self._running = True
    def terminate(self): 
      self._running = False   
    def run(self,msg,sleep): 
      while self._running:# and n > 0: 
        r = requests.post('https://1.x24.fr/a/bus.php',data=msg)
        time.sleep(sleep)#or n-=1

  def bgon():
    ''' envoyer des packets heartbeats au serveur de monitoring '''
    msg=GG('nbname')
    global c,t,threads;c=backgroundThread();t=threading.Thread(target=c.run,args=({'heartbeat':msg},60));t.daemon=True;t.start();threads+=[t];#c.terminate()

  def bgoff():
    ''' termination du monitoring sur demande '''
    msg=GG('nbname');requests.post('https://1.x24.fr/a/bus.php',data={'killhb':msg});global c;c.terminate()
    
def extractionMots3(x):
  ''' text cleanup : retrait des accents, harmonisation ponctuation, retrait de références, tags, hyperliens, caractères spéciaux '''
  import re
  import unidecode
  #x=x.lower()#uppercase is sign of agacement ou affirmation autoritaire
  x = unidecode.unidecode(x)
  x = re.sub('\n+', '.', x)#newline
  x = re.sub('\.+', '.', x)#single separator
  x=re.sub('@[a-zA-Z0-9\_]+','',x)#strip usernames
  x=re.sub('[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}\.[0-9]{1,3}','',x)#strip ips
  x=re.sub(':[a-zA-Z0-9\_]+','',x)#kindof
  x=re.sub('https?\S+(?=\s|$)','',x)#strip urls with negative lookahead :)
  notags=re.sub('<[^<]+?>', '', x)#suppression tags ouverture et fermeture ( conservant leur contenus interne : code, texte mis en forme, etc .. )
  noHTMLentities=re.sub('&[^;]{1,9};', '', notags)#cleanup &amp; &gt &lt
  stripped=re.sub("[^a-zA-Z0-9',.!\?]+",' ', noHTMLentities, flags=re.IGNORECASE)#autres que caractères de base  
  monospaced = re.sub('\s+', ' ', stripped)
  trimmed=monospaced.strip(' .')
  return trimmed
  #***************************
  lemitized=lemitizeWords(stripped)
  noStopWords=stopWordsRemove(lemitized)#i => retire du sens à la phrase conserve plupart mots clefs
  return trimAloneNumbers(noStopWords)    
  
def listGlobals():
  l=[]
  pk=list(globals().keys())
  for i in pk:
    if i.startswith('_'):
      pass;
    elif(type(globals()[i]) in [str,list,dict,np.array,pd.DataFrame]):
      l+=[i]
    elif(callable(globals()[i])):
      pass;
    else:
      pass;
      #p(i,type(globals()[i]))
  return l
# inject javascript function text2speech
#html("<script>lang=document.body.parentNode.getAttribute('lang');lang='en';function say(x,vid,p,rate){var defaultVoice=0,r=rate||1;if(lang.indexOf('fr')>-1){defaultVoice=2;r=1.8;}/*hortense*/vid=vid||defaultVoice;p=p||1;x=x||0;if(!x){x=alpow.getText();}if(!x)return;console.log(x);var y=new SpeechSynthesisUtterance(),voices=window.speechSynthesis.getVoices();y.voice=voices[vid];y.voiceURI='native';y.volume=1;y.rate=r;y.pitch=p;y.text=x;y.lang='en-US';speechSynthesis.speak(y);return x;}</script>")
