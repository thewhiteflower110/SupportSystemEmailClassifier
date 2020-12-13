def get_category(logits,t2):
    l=[]
    dicti = {0:"MDU",1:"Retirement",2:"Transfer"}

    for bert,lda in logits,t2:
        x=bert.index(min(bert)) #crossentropy loss is outputted in the tensor
        y=bert.index(max(lda))
    
        #assuming the order of the topics are same in bert and lda
        if x==y:
            l.append(dicti[x])
        else:
            l.append(dicti[y])
            