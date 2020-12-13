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

#def predict(app,data,bert=True,lda=True,combined=True):
def predict(app,data):

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
    
    bertmodel = bert.BertForSequenceClassification()
    bertmodel.load_state_dict(torch.load("./models/bert.bin"))
    logits=bert.predict(bertmodel, data)

    lda_model=gensim.models.ldamodel.LdaModel.load("./models/lda_train.model")
    t2=model_predict(lda_model,data)

    l['category']=lnm.get_category(logits,t2)
    df=pd.DataFrame(l)
    data=data.join(df)

    #format the responses
    reponses_data=[]
    for i in data:
        d={'filename':data['content'], "category": data['category']}
        #d1={'filename':"abc", "category": "MDU"}
        #d2={'filename':"def", "category": "Transfer"}
        #reponses_data.append(d1) 
        reponses_data.append(d) 

    response = app.response_class(response=flask.json.dumps(reponses_data), status=200, mimetype='application/json')
    return response

def train(filename,data):
    with ZipFile(filename, 'r') as zip: 
        zip.printdir()     
        zip.extractall("Zip_File/Train/") 

    dataset = pd.read_csv('/content/all_emails.csv')

    #extract data
    content_data=dataset['filtered_content']
    subject_data=dataset['subject']
    category_data=dataset['category']

    #call bert 
    num_labels=3
    batch_size=16
    config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
            num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
    optimizer_ft, criterion, exp_lr_scheduler = optim_config()
    dataloaders_dict, dataset_sizes=set_dataloaders(batch_size)
    #model = BertModel.from_pretrained('bert-base-uncased')
    model = bert.BertForSequenceClassification(num_lables)
    model_ft1 = bert.train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=10,dataloaders_dict, dataset_sizes)
    torch.save(model.state_dict(), "./models/bert.bin")

    #call lda
    number_topics = 3
    number_words = 3
    lda_model=model_fit(number_topics, number_words, subject_data)
    lda_model.save('./models/lda_train.model')
    return "SUCCESS"
