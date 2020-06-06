import pandas as pd
import numpy as np

def Label():
    train_aspects_label = pd.read_csv('./Data/twitter2015/train.tsv', sep='\t', header=0)
    train_aspects_label.insert(0, 'num', [x for x in range(len(train_aspects_label))])
    train_aspects_label['map'] = train_aspects_label['label'].map(str) + "#" + train_aspects_label['num'].map(str)
    train_aspects_label = train_aspects_label['map']
    train_aspects_label.to_csv('./Data/2015/train_aspects_label.txt', sep='\t', header=None, index=False)

    test_aspects_label = pd.read_csv('./Data/twitter2015/test.tsv', sep='\t', header=0)
    test_aspects_label.insert(0, 'num', [x for x in range(len(test_aspects_label))])
    test_aspects_label['map'] = test_aspects_label['label'].map(str) + "#" + test_aspects_label['num'].map(str)
    test_aspects_label = test_aspects_label['map']
    test_aspects_label.to_csv('./Data/2015/test_aspects_label.txt', sep='\t', header=None, index=False)

    val_aspects_label = pd.read_csv('./Data/twitter2015/val.tsv', sep='\t', header=0)
    val_aspects_label.insert(0, 'num', [x for x in range(len(val_aspects_label))])
    val_aspects_label['map'] = val_aspects_label['label'].map(str) + "#" + val_aspects_label['num'].map(str)
    val_aspects_label = val_aspects_label['map']
    val_aspects_label.to_csv('./Data/2015/val_aspects_label.txt', sep='\t', header=None, index=False)

def Image():
    train_image = pd.read_csv('./Data/twitter2015/train.tsv', sep='\t', header=0, usecols=['image'])
    train_image.to_csv('./Data/2015/train_image.txt', sep='\t', header=None, index=False )

    test_image = pd.read_csv('./Data/twitter2015/test.tsv', sep='\t', header=0, usecols=['image'])
    test_image.to_csv('./Data/2015/test_image.txt', sep='\t', header=None, index=False )

    val_image = pd.read_csv('./Data/twitter2015/val.tsv', sep='\t', header=0, usecols=['image'])
    val_image.to_csv('./Data/2015/val_image.txt', sep='\t', header=None, index=False )

def Aspect():
    train_aspects_text = pd.read_csv('./Data/twitter2015/train.tsv', sep='\t', header=0, usecols=['aspect'])
    train_aspects_text.to_csv('./Data/2015/train_aspects_text.txt', sep='\t', header=None, index=False )

    test_aspects_text = pd.read_csv('./Data/twitter2015/test.tsv', sep='\t', header=0, usecols=['aspect'])
    test_aspects_text.to_csv('./Data/2015/test_aspects_text.txt', sep='\t', header=None, index=False )

    val_aspects_text = pd.read_csv('./Data/twitter2015/val.tsv', sep='\t', header=0, usecols=['aspect'])
    val_aspects_text.to_csv('./Data/2015/val_aspects_text.txt', sep='\t', header=None, index=False )

def Text():
    train_text = pd.read_csv('./Data/twitter2015/train.tsv', sep='\t', header=0)
    train_text[['left','right']] = pd.DataFrame([x.split('$T$') for x in train_text['text'].tolist()])
    train_text['map'] = train_text['left'].map(str) + " " + train_text['aspect'].map(str) + " " + train_text['right'].map(str)
    train_text = train_text['map']
    train_text.to_csv('./Data/2015/train_text.txt', sep='\t', header=None, index=False )

    test_text = pd.read_csv('./Data/twitter2015/test.tsv', sep='\t', header=0)
    test_text[['left','right']] = pd.DataFrame([x.split('$T$') for x in test_text['text'].tolist()])
    test_text['map'] = test_text['left'].map(str) + " " + test_text['aspect'].map(str) + " " + test_text['right'].map(str)
    test_text = test_text['map']
    test_text.to_csv('./Data/2015/test_text.txt', sep='\t', header=None, index=False )

    val_text = pd.read_csv('./Data/twitter2015/val.tsv', sep='\t', header=0)
    val_text[['left','right']] = pd.DataFrame([x.split('$T$') for x in val_text['text'].tolist()])
    val_text['map'] = val_text['left'].map(str) + " " + val_text['aspect'].map(str) + " " + val_text['right'].map(str)
    val_text = val_text['map']
    val_text.to_csv('./Data/2015/val_text.txt', sep='\t', header=None, index=False )

if __name__ == '__main__':
    Image()
    Label()
    Aspect()
    Text()