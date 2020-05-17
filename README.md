#
- Usage : py toxic.py # -> runs the model in Flask on port 8082, accepts curl requests in body and returns the probability of this comment to be toxic ( recognizes languages : en, fr, ru, de ,es, pt, tk )
- Toxic : less than an insult, is a comment we might want to moderate before it lights up a spark
- Used in this competition https://www.kaggle.com/c/jigsaw-multilingual-toxic-comment-classification with roc_auc scoring of : 0.9163
- Using Xlm Roberta on Multiple TPU
- Training on 3.100.000 records with some of them translated by google API
