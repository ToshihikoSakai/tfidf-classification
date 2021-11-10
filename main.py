import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression


data = []        
# doccanoで出力したjsonlファイル
with open("train.jsonl") as f:
    for line in f:
        data.append(json.loads(line))

labels = []
texts = []
flag = 0

parse = []
# 事前に形態素解析したファイルを読み込む(スペース区切り)
with open("parsed.txt") as f:
    for line in f:
        parse.append(line)


for dic in data:
    #print(dic)
    #{'id': 9226, 'text': '', 'labels': []}
    texts.append(dic['text'])
    
    # labelsの中身を全てチェック
    for a in dic['labels']:
        #テキストへ
        text = ' '.join(map(str,a))
        #もし一つでも該当があれば
        if(('tagA' in text) or ('tagB' in text)):
            flag = 1
            break

    if(flag == 1):
        labels.append(1)
    else:
        labels.append(0)
        
    flag = 0

# textとlabelsの確認
for i in range(len(texts)):
    print(i)
    print(texts[i])
    print(labels[i])
    print("----------------")


# labelsが1になった数の確認
count = 0
for a in labels:
    if(a==1):
        count+=1

print(count)


## TF-IDF化
vectorizer = TfidfVectorizer()
X = vectorizer.fit_transform(texts)
# 未知データ予測のためのvocab取得
vocab = vectorizer.get_feature_names()

print(X.shape)

## SVMの処理
X_train, X_test, y_train, y_test = train_test_split(X, labels, train_size=0.9, random_state=0)

K = 10

method = "SVM"

scoring = {
        "p": "precision_macro",
        "r": "recall_macro",
        "f": "f1_macro",
        "a": "accuracy"
}


if(method=="SVM"):

    candidate_params = {
        'C': [1, 10, 100],
        'gamma': [0.01, 0.1, 1],
        'kernel' : ['linear','rbf']
    }
    
    classifiers = [
        svm.SVC(),
    ]
    
else:
    candidate_params = {
        'C': [0.01, 0.1, 1, 10, 100],
    }
    classifiers = [
        LogisticRegression(),
    ]

print("******** Running Trainning *******")
print("Num train examples = {}".format(X_train.shape))
print("Num test examples = {}".format(X_test.shape))
ML = "SVM" if(method=="SVM") else "logistic regression"
print("ML={}".format(ML))
  

for clas in classifiers:

    # classifiersインスタンス
    clf = clas
    print("----------------------------")
    print(clas)
    print("----------------------------")

    # GridSearch
    # cv = Kで層化KFoldになる
    gs = GridSearchCV(estimator=clf, param_grid=candidate_params, scoring='f1_macro',cv=K, n_jobs=-1)
    gs.fit(X_train, y_train)
    
    #　結果の表示
    print(gs.best_estimator_)
    print(gs.best_params_)

    # 混同行列の出力
    pred = gs.best_estimator_.predict(X_test)
    print(confusion_matrix(y_test,pred))
    print(classification_report(y_test,pred,digits=3))


    # 未知データの予測
    # 学習時のvocabularyをセット
    michi_vectorizer = TfidfVectorizer(vocabulary=vocab)
    # fitで得た語彙やidfを基に、文書をTF-IDF行列に変換する
    Y = michi_vectorizer.fit_transform(parse)
    # 予測
    pred = gs.best_estimator_.predict(Y)
    print(pred)

    with open('./data/classified_sentences.txt',mode='w') as ff:

        count = 0
        for a in range(len(parse)):
            if(pred[a] == 1):
                ff.write(''.join(parse[a].split()))
                ff.write('\n')
                count += 1
        print("{}/{}".format(count,len(parse)))

