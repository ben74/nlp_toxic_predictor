'''
#usage : run flask
py toxic.py &
#then
curl -k 127.0.0.1:8082 -F "input=unexpected"
curl -k 127.0.0.1:8082 -F "text=Its okay to be a gay black woman"
#latest checkpoint : wget http://1.x24.fr/a/jupyter/poc7/p8_p2xlmr192Fast_NotEn_17_lastCheckpoint.h5.zip
'''

#mainframe inclusion
if 'MainFrame':
    import os
    import sys
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    #sys.stdout = open(os.devnull, "w")
    #sys.stderr = open(os.devnull, "w")
    #}{
    requiredModules='Flask tensorflow transformers pysftp numpy requests unidecode'.split(' ')
    if True:
      fn='modules-versions.txt'
      os.system('pip freeze > '+fn)
      #ftpput(fn)
      installed=''
      with open(fn) as f:
        installed += f.read()

      for module in requiredModules:
        if(module+'==' not in  installed):
          #print('Trying to install :',module)
          os.system('pip install '+module)

      os.system('pip freeze > '+fn);

    #os.system('rm -f alpow.py gv.py');os.system('wget https://alpow.fr/gv.py');os.system('wget https://alpow.fr/alpow.py');
    import gv;import alpow;from alpow import *
    SG('noClassRewrite',False);
    SG('webRepo','https://1.x24.fr/a/jupyter/');SG('sftp',{'cd':'poc7','web':GG('webRepo'),'h':'-','u':'-','p':'-'});SG('useFTP',False);#ReadOnly

    import tensorflow as tf

    def load(fn='allVars',onlyIfNotSet=1):
      fns=fn.split(',')
      for fn in fns:
        fn=fn.strip(', \n')
        ok=1
        if(len(fn)==0):
          continue
        if(onlyIfNotSet):
          if fn in globals().keys():
      #override empty lists, dict, dataframe and items
            if type(globals()[fn])==type:
              continue;
            elif type(globals()[fn])==pd.DataFrame:
              if globals()[fn].shape[0]>0:
                continue
            elif(type(globals()[fn])==dict):
              if(len(globals()[fn])>0):
                continue
            elif(type(globals()[fn])==list):
              if(len(globals()[fn])>0):
                continue
            elif(type(globals()[fn])==scipy.sparse.csr.csr_matrix):
              if(globals()[fn].shape[0]>0):
                continue
            elif(type(globals()[fn])==np.ndarray):
              if(globals()[fn].shape[0]>0):
                continue
      #si déjà définie, passer au prochain
            elif(globals()[fn]):
              continue
        globals().update(alpow.resume(fn))
      #endfor fn
      return;

    #load('X_train')

    def extract(x):
      liste=list(x.keys())
      for i in liste:
        globals()[i]=x[i]
      p('extracted : ',','.join(liste))

    #jeuDonnees=compact('y_test,')
    def compact(variables):
      x={}
      for i in variables.split(','):
        x[i]=globals()[i]
      p('compacted : ',variables)
      return x

    def loadIfNotSet(x):
      if x not in globals().keys():
        load(x)

    def isToxic(x):
        return model.predict(s)

    ###############}{
    import numpy as np
    import warnings,psutil
    #ftpls()

if 'project specific':
    from transformers import TFAutoModel, AutoTokenizer
    strategy = tf.distribute.get_strategy()

    def getModel(mdlname,mdl,inputLen,loss='binary_crossentropy',nbOut=1,dense=0,freeze=0):
      import tensorflow as tf
      @tf.function( experimental_relax_shapes=True )
      def f(x):
        return x
      from tensorflow.keras.layers import Dense, Input
      from tensorflow.keras.optimizers import Adam
      from tensorflow.keras.models import Model
      from transformers import TFAutoModel, AutoTokenizer
      from tensorflow.keras.models import Sequential
      from tensorflow.keras.layers import Dropout, Flatten, Dense, GlobalAveragePooling2D, Conv2D, MaxPooling2D

      with strategy.scope():
        transformer_layer = TFAutoModel.from_pretrained(mdl)#large is > base
        input_word_ids = Input(shape=(inputLen,), dtype=tf.int32, name="input_word_ids")
        sequence_output = transformer_layer(input_word_ids)[0]
        cls_token = sequence_output[:, 0, :]
        if dense>0:#
          dense_layer = Dense(dense, activation='relu')(cls_token);
          out = Dense(nbOut, activation='sigmoid')(dense_layer)#32 nuances#bcp plus lourd à entrainer !
        else:
          out = Dense(nbOut, activation='sigmoid')(cls_token)#, dtype='float16'
        mdl = Model(inputs=input_word_ids, outputs=out)
        mdl._name=mdlname
        if type(freeze) is not int:
          for layer in mdl.layers:
            if(type(layer)==freeze):
              p(layer,'not trainable')
              layer.trainable = False

        mdl.compile(tf.keras.optimizers.Adam(lr=1e-5),loss=loss, metrics=[tf.keras.metrics.AUC(),'accuracy'])

      return mdl

    ml=192
    mdlname='p8_p2xlmr192Fast_NotEn_17'
    mdl='jplu/tf-xlm-roberta-base'
    tokenizer = AutoTokenizer.from_pretrained(mdl)

    bestTextEncoding=17
    def TraitementTexte(x,mod):
        '''
        TraitementTexte(phrase,MultiplicationDeChiffresPremiersCiDessous)
        #5*7*11*13*17 supposé être le meilleur
        mais 17 en réalité
        '''
        if(mod==1):
            return x
        import re
        import unidecode
        #x=#uppercase is sign of agacement ou affirmation autoritaire
        if(mod%2==0):
            x=unidecode.unidecode(x)
        if(mod%3==0):
            x=x.lower()
        if(mod%5==0):
            x=re.sub('\s+', ' ', x)
        if(mod%7==0):
            x=re.sub('https?\S+(?=\s|$)','',x)
        if(mod%11==0):
            x=re.sub('<[^<]+?>', '', x)#clear innter HTML tags
        if(mod%13==0):
            x=re.sub('&[^;]{1,9};', '', x)#htmlentitites
        if(mod%17==0):#tutti
            x=re.sub("[^a-zA-Z0-9',.!\?]+",' ', x, flags=re.IGNORECASE)
        if(mod%19==0):
            x=x.strip(' .')
        #23,29,31,37,41,
        return x;

    def regular_encode(texts, tokenizer, maxlen=512):
        enc_di = tokenizer.batch_encode_plus(
            texts,
            return_attention_masks=False,
            return_token_type_ids=False,
            pad_to_max_length=True,
            max_length=maxlen
        )

        return np.array(enc_di['input_ids'])

    def encode(x):
        x=TraitementTexte(x,bestTextEncoding)
        x=regular_encode([x], tokenizer, maxlen=ml)
        #p(x.shape)#1,192
        return x

    def isToxic(x):
        x2=encode(x)
        pred=model.predict(x2)
        res=pred[0][0]
        p("\n",x,'=>',pred,'=>',res)
        return res

    #p(encode("I wish you'll be dead you gay motherfucker"))

    model = getModel(mdlname, mdl, ml)
    weights='p8_p2xlmr192Fast_NotEn_17_lastCheckpoint.h5'
    
    if not os.path.exists(weights):
        if not os.path.exists(weights+'.zip'):
            import requests
            r=requests.get('http://1.x24.fr/a/'+weights+'.zip',stream=True)
            with open(weights+'.zip','wb') as f:
                f.write(r.raw.read())    
                
        p('unzip once')
        os.system('unzip '+weights+'.zip');
        
    model.load_weights(weights)
    
    clear()
    p('_'*180)
    isToxic("I wish you'll be dead you gay motherfucker")
    isToxic('ceci est une phrase comportant des fleurs et des abeilles :)')
    isToxic('I am a gay black woman')

#}{flask mode#******
if 'flask':#len(sys.argv)<2:
    from flask import Flask, render_template, url_for, request
    app = Flask(__name__)
    @app.route('/')
    def home():
        return """<html><head><title>Nlp Toxic Comment Prediction</title>
<style>html{font-size:10px;background:#000 url('//x24.fr/0/b1.jpg##20200103SkiRandoClusaz') top center fixed;}/*background-size:110%;background-repeat:repeat;*/
body{height:100vh;color:#FFF;margin:0;}
body,pre{font:2rem 'Dancing Script',Assistant,roboto,calibri,corbel,verdana;}
*{transition:all .5s}
a{color:#0F0;}
a:hover{color:#FC0;}

fieldset{margin:0 2vw;background:rgba(255,255,255,0.1);}
legend{padding:0 3rem;}/*margin:auto;*/

input,textarea{width:100%;padding-left:1rem;font-size:2rem}
textarea{height:30vh;}

input[type=submit]{
    cursor:pointer;height:5vh;font-size:4rem;
}
input[type=submit]:hover{filter:invert(100%);}
h1,h2{color:#F00;margin:0;}

</style>
</head><body><center><fieldset><legend>Toxic Text Nlp Predictor</legend><form method='post' action='/'><textarea name='text'></textarea><br><input type=submit value=submit></form></fieldset></center></body></html>
""";

    @app.route('/', methods=['POST'])
    def post():
        if (request.method == 'POST'):
        #print(('filepath' in request.form.keys()))
            if 'text' in request.form.keys():
                if(len(request.form['text'])>1):
                    return str(isToxic(request.form['text']))
        return 'ko'
    app.run(host='0.0.0.0', port=8082, debug=True)
    p('end')
