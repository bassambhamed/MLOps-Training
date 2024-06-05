from sklearn.metrics import precision_recall_fscore_support as score

def evaluate_model(actual,pred):
    precision,recall,fscore,support=precision,recall,fscore,support=score(actual,pred,average='macro')
    return precision,recall,fscore,support