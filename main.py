# get the data
from flask import json
from zipfile import ZipFile
import flask
import pandas as pd
import bert
import lda
import lnm
import torch.nn.functional as F
from zipfile import ZipFile
from pytorch_pretrained_bert import BertConfig
import torch
import gensim
import os
import extract_msg
import sys

# def predict(app,data,bert=True,lda=True,combined=True):


def zip2csv(root_folders, CSV_PATH):  # root_folders = [f1,f2]

    d = {'category': [], 'subject': [], 'content': []}
    print("Root folder", root_folders)
    print("CSV_PATH --------------")
    print(CSV_PATH)
    for i in root_folders:
        # print(os.getcwd())
        # i="C:\Users\Juhi Kamdar\Desktop\iitb\SupportSystemEmailClassifier\Zip_File\Predict\Test_Data_-_Eliminations"
        l = os.listdir(i)
        for j in l:
            f = os.path.join(i, j)
            if f[-3:] == "msg":
                msg = extract_msg.Message(f)
                d['category'].append(i.split('/')[-1])
                d['subject'].append(msg.subject)
                d['content'].append(msg.body)

    all_emails = pd.DataFrame(d)
    d = {'filtered_content': []}

    for i in all_emails['content']:
        i = i.split('\r\n')
        count = 0
        for ij, j in enumerate(i):
            if 'Subject' in j:
                count += 1
            elif 'Re:' in j:
                i.remove(j)
            elif 'wrote:' in j:
                i.remove(j)
        for z in range(count):
            for ij, j in enumerate(i):
                if 'From:' in j:
                    start = ij
                    for ik, k in enumerate(i[ij:]):
                        if 'Subject:' in k:
                            end = ik+ij
                            break
                    break
            del i[start:end+2]
        if ' ' in i:
            i.remove(" ")
        filtered = ' '.join(i)
        # print(filtered.strip())
        d['filtered_content'].append(filtered.strip())

    df = pd.DataFrame(d)
    all_emails = all_emails.join(df)
    all_emails.to_csv(CSV_PATH)


def predict(app, zip1):

    #zip1="C://Users//Juhi Kamdar//Desktop//iitb//SupportSystemEmailClassifier//Zip_File//Test_Data_-_Eliminations.zip"
    print('ZIP1 -------')
    print(zip1)
    with ZipFile('./Zip_File/' + zip1, 'r') as zip:
        print("hi zip")
        zip.printdir()
        zip.extractall("./Zip_File/Predict/")

    s = "./Zip_File/Predict/Test Data - Eliminations/"
    root_folders = [s]
    CSV_PATH = "./predict.csv"
    zip2csv(root_folders, CSV_PATH)
    # CSV_PATH="./emails(1).csv"
    script_dir = os.path.abspath(os.path.dirname(sys.argv[0]) or '.')
    print("SCRIPT DIR ________")
    print(script_dir)
    data = pd.read_csv(script_dir + '/predict.csv')
    num_labels = 3

    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    bertmodel = bert.BertForSequenceClassification(num_labels, config)
    bertmodel.load_state_dict(torch.load(
        "./SupportSystemEmailClassifier/models/bert.bin"))
    print('bert loaded')
    logits = bert.predict(bertmodel, data['filtered_content'])
    print("Hi! bert predicted!")

    print("Hi! you entered predict")
    lda_model = gensim.models.ldamodel.LdaModel.load(
        "./models/lda_train.model")
    print("Hi! lda_model")
    print(data['filtered_content'])
    t2 = lda.model_predict(lda_model, data['subject'])
    print("Hi! t2")
    l = {}
    '''
    logits = [[1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], [1, 2, 3], [4, 5, 6], [7, 8, 9], [
        10, 11, 12], [13, 14, 15], ]
    '''
    l['category'] = lnm.get_category(logits, t2)
    df = pd.DataFrame(l)
    # print("PREPREPREPREPREPRERPERPDFDFDFDFDFDFDFDFFDFDFDFDFDFFDFDFDFFDFD")
    # print(data)
    data = data.drop(columns='category')
    data = data.join(df)
    print("DFDFDFDFDFDFDFDFFDFDFDFDFDFFDFDFDFFDFD")
    print(data)

    # format the responses
    reponses_data = []
    print("LENGTHHHHHHHHHHHH!@#$%^&*_!@#$%^&*", len(data[:]))
    for i in range(len(data[:])):
        # print()
        d = {'filename': data['subject'][i], "category": data['category'][i]}
        reponses_data.append(d)
    print("respones_data", type(reponses_data), reponses_data)
    response = app.response_class(response=flask.json.dumps(
        reponses_data), status=200, mimetype='application/json')
    response.headers.add('Access-Control-Allow-Origin', '*')
    return response


def train(filename, data, learning_rate, num_epochs):

    with ZipFile(filename, 'r') as zip:
        zip.printdir()
        zip.extractall("Zip_File/Train/")
    CSV_PATH = "./data/all_emails.csv"
    root_folders = ["./Zip_File/Train/"]  # add if more folders
    zip2csv(root_folders, CSV_PATH)

    dataset = pd.read_csv('CSV_PATH')

    # extract data
    content_data = dataset['filtered_content']
    subject_data = dataset['subject']
    category_data = dataset['category']

    # call bert
    num_labels = 3
    batch_size = 16
    # num_epochs=2
    # learning_rate=0.00001
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
                        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    optimizer_ft, criterion, exp_lr_scheduler = optim_config(learning_rate)
    dataloaders_dict, dataset_sizes = set_dataloaders(batch_size)
    #model = BertModel.from_pretrained('bert-base-uncased')
    model = bert.BertForSequenceClassification(num_lables)
    model_ft1 = bert.train_model(model, criterion, optimizer_ft,
                                 exp_lr_scheduler, num_epochs, dataloaders_dict, dataset_sizes)
    torch.save(model.state_dict(), "./models/bert.bin")

    # call lda
    number_topics = 3
    number_words = 3
    lda_model = model_fit(number_topics, number_words, subject_data)
    lda_model.save('./models/lda_train.model')
    return "true"
