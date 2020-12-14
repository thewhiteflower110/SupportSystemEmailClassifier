#get the data
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

#def predict(app,data,bert=True,lda=True,combined=True):
def zip2csv(root_folders,CSV_PATH): #root_folders = [f1,f2]
    d={'category':[],'subject':[],'content':[]}

    for i in root_folders:
    l=os.listdir(i)
    for j in l:
        f=os.path.join(i,j)
        if f[-3:]=="msg":
        msg = extract_msg.Message(f)
        d['category'].append(i.split('/')[-1])
        d['subject'].append(msg.subject)
        d['content'].append(msg.body)

    all_emails=pd.DataFrame(d)
    d={'filtered_content':[]}
    
    for i in all_emails['content']:
        i=i.split('\r\n')
        count=0
        for ij,j in enumerate(i):
            if 'Subject' in j:
            count+=1
            elif 'Re:' in j:
            i.remove(j)
            elif 'wrote:' in j:
            i.remove(j)
        for z in range(count):
            for ij,j in enumerate(i):
            if 'From:' in j:
                start=ij
                for ik,k in enumerate(i[ij:]):
                if 'Subject:' in k:
                    end=ik+ij
                    break
                break
            del i[start:end+2]
        if ' ' in i:
            i.remove(" ")
        filtered=' '.join(i)
        #print(filtered.strip())
        d['filtered_content'].append(filtered.strip())

  df=pd.DataFrame(d)
  all_emails=all_emails.join(df)
  all_emails.to_csv(CSV_PATH)

def predict(app,data):
    '''
    if bert==True:
        bertmodel = bert.BertForSequenceClassification()
        bertmodel.load_state_dict(torch.load("./models/bert.bin"))
        logits=bert.predict(bertmodel, data)
        print(torch.nn.softmax(logits))

    if lda==True:
        #prediction here
        lda_model=gensim.models.ldamodel.LdaModel.load("./models/lda_train.model")
        model_predict(lda_model,data)
    
    if combined==True:
        bertmodel = bert.BertForSequenceClassification()
        bertmodel.load_state_dict(torch.load("./models/bert.bin"))
        logits=bert.predict(bertmodel, data)

        lda_model=gensim.models.ldamodel.LdaModel.load("./models/lda_train.model")
        t2=model_predict(lda_model,data)

        model=lnm.LNM()
        model.load_state_dict(torch.load("./model/combined"))
        lnm.predict(model,logits,t2)
    '''
    bertmodel = bert.BertForSequenceClassification()
    bertmodel.load_state_dict(torch.load("./models/bert.bin"))
    logits=bert.predict(bertmodel, data['content'])

    lda_model=gensim.models.ldamodel.LdaModel.load("./models/lda_train.model")
    t2=lda.model_predict(lda_model,data['content'])
    l={}
    l['category']=lnm.get_category(logits,t2)
    df=pd.DataFrame(l)
    data=data.join(df)

    #format the responses
    reponses_data=[]
    for i in data:
        d={'filename':data['content'][i], "category": data['category'][i]}
        reponses_data.append(d) 

    response = app.response_class(response=flask.json.dumps(reponses_data), status=200, mimetype='application/json')
    return response

def train(filename,data,learning_rate,num_epochs):
    with ZipFile(filename, 'r') as zip: 
        zip.printdir()     
        zip.extractall("Zip_File/Train/") 
    
    CSV_PATH="./data/all_emails.csv"
    root_folders=["./Zip_File/Train/"] #add if more folders
    zip2csv(root_folders,CSV_PATH)

    dataset = pd.read_csv('CSV_PATH)

    #extract data
    content_data=dataset['filtered_content']
    subject_data=dataset['subject']
    category_data=dataset['category']

    #call bert 
    num_labels=3
    batch_size=16
    #num_epochs=2   
    #learning_rate=0.00001
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    optimizer_ft, criterion, exp_lr_scheduler = optim_config(learning_rate)
    dataloaders_dict, dataset_sizes=set_dataloaders(batch_size)
    #model = BertModel.from_pretrained('bert-base-uncased')
    model = bert.BertForSequenceClassification(num_lables)
    model_ft1 = bert.train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs,dataloaders_dict, dataset_sizes)
    torch.save(model.state_dict(), "./models/bert.bin")

    #call lda
    number_topics = 3
    number_words = 3
    lda_model=model_fit(number_topics, number_words, subject_data)
    lda_model.save('./models/lda_train.model')
    return "true"
