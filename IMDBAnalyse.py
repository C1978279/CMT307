#---IMDbAnalyse.py--#
#Import Packages
import numpy as np
import sklearn
import nltk
lemmatizer=nltk.stem.WordNetLemmatizer()#initialize lemmatization functions
#--Setup stopwords
nltk.download('stopwords')
stopwords=set(nltk.corpus.stopwords.words('english'))
stopwords=stopwords.union({".",",","'","<",">","br"})
#--Setup Part-of-speech
nltk.download('averaged_perceptron_tagger')

#--Training Data 
path_train_pos='datasets/IMDb/train/imdb_train_pos.txt'
path_train_neg='datasets/IMDb/train/imdb_train_neg.txt'
#--Development Data
path_dev_pos='datasets/IMDb/dev/imdb_dev_pos.txt'
path_dev_neg='datasets/IMDb/dev/imdb_dev_neg.txt'
#--Test Data
path_test_pos='datasets/IMDb/test/imdb_test_pos.txt'
path_test_neg='datasets/IMDb/test/imdb_test_neg.txt'

def open_reviews(path_pos,path_neg):
	with open(path_pos,encoding="utf-8") as f:
		data_pos=f.readlines()
	with open(path_neg,encoding="utf-8") as ff:
		data_neg=ff.readlines()
	#--Create list of all positive and negative reviews, separate text from Y value
	reviews=[x.strip() for x in data_pos+data_neg]
	Y=list(('1'*len(data_pos))+('0'*len(data_neg)))
	Y=np.asarray([int(x) for x in Y])
	return reviews,Y

def text_tokens(text):
	word_list=[]
	text_sent=nltk.tokenize.sent_tokenize(text)#break text into sentences
	for sent in text_sent:
		list_token_sent=nltk.tokenize.word_tokenize(sent)#break sentences into words
		for word in list_token_sent:
			word_list.append(lemmatizer.lemmatize(word).lower())#lemmatize and convert to lowercase
	return word_list

def merge_sublists(list):
	final=[]
	for item in list:
		final+=item
	return final

def text_pos(text):
	#--Create list of all Parts-of-Speech in text
	sent_pos=nltk.tag.pos_tag_sents([nltk.tokenize.word_tokenize(x)\
		for x in nltk.tokenize.sent_tokenize(text)])
	merge_pos=merge_sublists(sent_pos)
	return_pos=[x[1] for x in merge_pos]
	return return_pos

def get_feat_list(text_array,num_feats):
	D={}
	P={}
	for text in text_array:
		#--Create dictionary of Words
		word_list=text_tokens(text)
		for item in word_list:
			if item in stopwords:continue
			if item not in D: D[item]=1
			else: D[item]+=1
		#--Create dictionary of Parts-of-Speech
		pos_list=text_pos(text)
		for item2 in pos_list:
			if item2 not in P: P[item2]=1
			else: P[item2]+=1
	top_D=[x for x in sorted(D.items(),key=lambda x:x[1],reverse=True)][:num_feats]
	top_P=[x for x in sorted(P.items(),key=lambda x:x[1],reverse=True)]
	return top_D, top_P

def feat_ext(text,D,P):
	v_freq=np.zeros(len(D));v_num_sent=np.zeros(1);v_pos=np.zeros(len(P))
	#frequency of words
	text_word_list=text_tokens(text)
	for i,word in enumerate(D):
		if word in text_word_list:
			v_freq[i]=text_word_list.count(word)
	#number of sentences
	v_num_sent[0]=len(nltk.tokenize.sent_tokenize(text))
	#frequency of parts of speech
	text_pos_list=text_pos(text)
	for j,pos in enumerate(P):
		if pos in text_pos_list:
			v_pos[j]=text_pos_list.count(pos)
	#concatenate all features
	v_feat=np.concatenate((v_freq,v_num_sent,v_pos))
	return v_feat

def train_classify(text_array,Y,D,P):
	X_train=[]
	for text in text_array:
		v_text=feat_ext(text,D,P)#feature extraction
		X_train.append(v_text)
	svm_clf=sklearn.svm.SVC(kernel="linear",gamma="auto")#initialize svm classifier
	svm_clf.fit(np.asarray(X_train),Y)
	return svm_clf

def performance(Y_gold,Y_predict, av="macro"):
	a=sklearn.metrics.accuracy_score(Y_gold,Y_predict)
	p=sklearn.metrics.precision_score(Y_gold,Y_predict, average=av)
	r=sklearn.metrics.recall_score(Y_gold,Y_predict, average=av)
	f=sklearn.metrics.f1_score(Y_gold,Y_predict, average=av)
	return (a,p,r,f)

#--Read in training/development/test data
reviews_train,Y_train=open_reviews(path_train_pos,path_train_neg)
reviews_dev,Y_dev=open_reviews(path_dev_pos,path_dev_neg)
reviews_test,Y_test=open_reviews(path_test_pos,path_test_neg)
print("Files opened")
#---Feature Selection using Development Set
number_features=list(range(200,1100,100))
high_acc=0.0
for num_feat in number_features:
	#--Train SVM classifier
	D,P=get_feat_list(reviews_train,num_feat)
	svm_clf=train_classify(reviews_train,Y_train,D,P)
	X_dev=[]
	for text_dev in reviews_dev:
		v_text_dev=feat_ext(text_dev,D,P)
		X_dev.append(v_text_dev)
	#--Perform testing on deevelopment set
	Y_dev_pred=svm_clf.predict(np.asarray(X_dev))
	acc_dev=sklearn.metrics.accuracy_score(Y_dev,Y_dev_pred)
	#--Record best development test results
	if acc_dev>high_acc:
		high_acc=acc_dev
		best_num_features=num_feat
		best_feat=[D,P]
		best_svm=svm_clf
	print("Development test complete:",num_feat,"features")
print("Highest accuracy on development set:",high_acc,"with",best_num_features,"features")
#--Test model and performance
X_test=[]
for text_test in reviews_test:
	v_text_test=feat_ext(text_test,best_feat[0],best_feat[1])
	X_test.append(v_text_test)
Y_test_pred=best_svm.predict(np.asarray(X_test))
#--Calculate performance
results=performance(Y_test,Y_test_pred)
#--Return results in LaTeX format
print("{ind}&${acc:.2f}\\%$&${prec:.2f}\\%$&${rec:.2f}\\%$&${f1:.2f}\\%$\\\\".format(\
			ind="Test Set",acc=results[0]*100,prec=results[1]*100,rec=results[2]*100,f1=results[3]*100))