#get the data
import pandas as pd
import bert
import lda
dataset = pd.read_csv('/content/all_emails.csv')

#extract data
content_data=dataset['filtered_content']
subject_data=dataset['subject']
category_data=dataset['category']

#call bert 
num_labels=3
batch_size=16
predict_text="juhi pension"
config = BertConfig(vocab_size_or_config_json_file=32000, hidden_size=768,
        num_hidden_layers=12, num_attention_heads=12, intermediate_size=3072)
optimizer_ft, criterion, exp_lr_scheduler = optim_config()
dataloaders_dict, dataset_sizes=set_dataloaders(batch_size)
#model = BertModel.from_pretrained('bert-base-uncased')
model = bert.BertForSequenceClassification(num_lables)
model_ft1 = bert.train_model(model, criterion, optimizer_ft, exp_lr_scheduler,num_epochs=10,dataloaders_dict, dataset_sizes)

#insert text here
logits=bert.predict(model_ft1, predict_text)

#call lda
number_topics = 3
number_words = 3
lda_model=model_fit(number_topics, number_words, subject_data)
#with open('/content/outfile', 'wb') as fp:
#    pickle.dump(lda_model, fp)

model_predict(lda_model,prints=True,data_samples)