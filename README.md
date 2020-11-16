---
- Usage : py toxic.py # -> runs the model in Flask on port 8082, accepts curl requests in body and returns the probability of this comment to be toxic ( recognizes languages : en, fr, ru, it ,es, pt, tk )
- Then use either main interface at {{ http://127.0.0.1:8082 }} or perform Ajax or curl requests like this : {{ curl -k 127.0.0.1:8082 -F "text=put whatever you like either some insulting or discriminatory speech or something not toxic at all" }}
---
- Toxicity : more than irony, less than an insult, is a comment we might want to moderate before it lights up a spark
- Used in this competition https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification with roc_auc scoring of : 0.9163
- Using Xlm Roberta on Multiple TPU
- Trained on 3.100.000 records with some of them translated by google API

![visitors](https://visitor-badge.glitch.me/badge?page_id=gh:ben74:nlpToxicPredictor)
