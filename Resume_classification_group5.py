import docx2txt
import streamlit as st
import pdfplumber
import re 
from nltk.tokenize import RegexpTokenizer 
from nltk.stem import  WordNetLemmatizer 
from nltk.corpus import stopwords
stop=set(stopwords.words('english'))
from pickle import load

model=load(open(r"C:\\Users\\HP\\Downloads\\finalized_model.sav",'rb'))
vectors=load(open(r"C:\\Users\\HP\\Downloads\\vectorizer.sav",'rb'))

resume=[]

def display(doc_file):
    if doc_file.type=="application/vnd.openxmlformats-officedocument.wordprocessingml.document":
        resume.append(docx2txt.process(doc_file))
    else:
        with pdfplumber.open(doc_file) as pdf:
            pages=pdf.pages[0]
            resume.append(pages.extract_text())
    return resume
            

def preprocess(sentence):
    sentence1=str(sentence)
    sentence2=sentence1.lower()
    sentence3=sentence2.replace('{html}',"")
    cleanr=re.compile('<.*?>')
    cleantext=re.sub(cleanr,'',sentence3)
    rem_url=re.sub(r'http\S+','',cleantext)
    rem_num=re.sub('[0-9]+','',rem_url)
    tokenizer=RegexpTokenizer(r'\w+')
    tokens=tokenizer.tokenize(rem_num)
    filtered_words=[w for w in tokens if len(w)>2 if not w in stopwords.words('english')]
    lemmatizer=WordNetLemmatizer()
    lemma_words=[lemmatizer.lemmatize(w) for w in filtered_words]
    return" ".join(lemma_words)
    
def main():
    st.title("Resume Classification Using NLP")
    menu=["Classifier"]
    choice = st.sidebar.selectbox("Menu",menu)
    
    if choice=="Classifier":
        upload_file=st.file_uploader('Upload a resume file', type=['docx','pdf'], accept_multiple_files=True)
        
        if st.button('Predict'):
            for doc_file in upload_file:
                if doc_file is not None:

                    displayed=display(doc_file)
                    cleaned=preprocess(displayed)
                    predicted=model.predict(vectors.transform([cleaned]))
                
                    if int(predicted)==0:
                        st.success("PeopleSoft Resume")
                    elif int(predicted)==1:
                        st.success("ReactJS Developer Resume")
                    elif int(predicted)==2:
                        st.success("SQL Developer Resume")
                    else:
                        st.success("Workday Resume")

    
if __name__ == '__main__':
	main()   
    