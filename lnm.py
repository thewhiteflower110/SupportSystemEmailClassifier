def get_category(logits,t2):
    l=[]
    dicti_bert = {0:"Transfer",1:"Retirement",2:"MDU"}
    dicti_lda = {0:"Transfer",1:"MDU",2:"Retirement"}

    for bert,lda in logits,t2:
        x=bert.index(min(bert)) #crossentropy loss is outputted in the tensor
        y=bert.index(max(lda))
    
        #assuming the order of the topics are same in bert and lda
        if dicti_bert[x]==dicti_lda[y]:
            l.append(dicti_bert[x])
        else:
            l.append(dicti_lda[y])            
