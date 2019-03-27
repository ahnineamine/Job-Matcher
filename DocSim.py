import numpy as np
import spacy
from scipy.spatial import distance
import gensim
from gensim.corpora import Dictionary
from gensim.models.tfidfmodel import TfidfModel
from gensim import corpora, models, similarities
from gensim.matutils import sparse2full
from gensim.parsing.preprocessing import preprocess_documents
import math

class DocSim(object):
    def __init__(self, w2v_model):
        self.w2v_model = w2v_model
    
    def _keep_token(self,t):
        return (t.is_alpha and 
            not (t.is_space or t.is_punct or 
                    t.is_stop or t.like_num))
    
    def _lemmatize_doc(self,doc):
        return [ t.lemma_ for t in doc if self._keep_token(t)]
    
    
      #Gensim to create a dictionary and filter out stop and infrequent words (lemmas).
    def _get_docs_dict(self, docs):
        docs_dict = Dictionary(docs)
        #CAREFUL: For small corpus carefully modify the parameters for filter_extremes.
        #docs_dict.filter_extremes(no_below=5, no_above=0.2)
        docs_dict.compactify()
        return docs,docs_dict
    
    def _gensim_preprocess(self,document):
       #preprocess the documents using gensim
        proc_result=[]
        for c in document :
            doc=gensim.utils.simple_preprocess(c)
            proc_result.append(doc)
        return proc_result  
    
    #def _preprocess(self, doc_list):
        #Load spacy model
        #nlp  = spacy.load('en')
        #lemmatise docs
        #docs = [self._lemmatize_doc(nlp(doc)) for doc in doc_list] 
        #docs = self.gensim_preproc(doc_list)
        #Get docs dictionary
        #docs_dict = self._get_docs_dict(doc_list)
        #return docs_dict

    def vector(self,word):
        try:
            vec = self.w2v_model[word]
            return vec
        except KeyError:
            # Ignore, if the word doesn't exist in the vocabulary
            return [0 for i in range(150)]
    
    
    def _get_tfidf(self, docs, docs_dict):
        docs_corpus = [docs_dict.doc2bow(doc) for doc in docs]
        model_tfidf = TfidfModel(docs_corpus, id2word=docs_dict)
        docs_tfidf  = model_tfidf[docs_corpus]
        docs_vecs   = np.vstack([sparse2full(c, len(docs_dict)) for c in docs_tfidf])
        return docs_vecs
    
    def tfidf_weighted_wv(self,docs,docs_dict):
        #tf-idf
        tfidf_w   = self._get_tfidf(docs,docs_dict)
        #Load word2vec vectors
        tfidf_emb_vecs = np.vstack([self.vector(docs_dict[i]) for i in range(len(docs_dict))])
        #we just need to matrix multiply docs_vecs with tfidf_emb_vecs
        docs_emb = np.dot(tfidf_w, tfidf_emb_vecs)

        return docs_emb


    def _cosine_sim(self, vecA, vecB):
        """Find the cosine similarity distance between two vectors."""
        csim = np.dot(vecA, vecB) / (np.linalg.norm(vecA) * np.linalg.norm(vecB))
        if np.isnan(np.sum(csim)):
            return 0
        return csim

    def Euclidean(self,vec1, vec2) :
        return distance.euclidean(vec1, vec2)

    def Theta(self,vec1, vec2) :
        return math.acos(self._cosine_sim(vec1,vec2)) + math.radians(10)

    def Triangle(self,vec1, vec2) :
        theta = math.radians(self.Theta(vec1,vec2))
        return (np.linalg.norm(vec1) *np.linalg.norm(vec2) * math.sin(theta)) / 2

    def Magnitude_Difference(self,vec1, vec2) :
        return abs(np.linalg.norm(vec1) - np.linalg.norm(vec2))

    def Sector(self,vec1, vec2) :
        ED = self.Euclidean(vec1, vec2)
        MD = self.Magnitude_Difference(vec1, vec2)
        theta = self.Theta(vec1, vec2)
        return math.pi * math.pow((ED+MD),2) * theta/360

    def TS_SS(self,vec1, vec2) :
        """Find the Triangle Similarity - Sector Similarity between two vectors."""
        return self.Triangle(vec1, vec2) * self.Sector(vec1, vec2)


    def calculate_similarity(self, target_docs=[],L_index=[]):
        """Calculates & returns similarity scores between given source document "job offer" & all
        the target documents "CVs". """
        if isinstance(target_docs, str):
            target_docs = [target_docs]
        target = self._gensim_preprocess(target_docs)
        target_tuple = self._get_docs_dict(target) 
        docs_emb = self.tfidf_weighted_wv(target_tuple[0],target_tuple[1])
        without_source_emb = docs_emb[:-1]
        source_emb = docs_emb[-1]
        results = []
        for elem,index in zip(without_source_emb,L_index):
            sim_score = self.TS_SS(elem, source_emb)
            #sim_score = self._cosine_sim(elem, source_emb)
            #if sim_score > threshold:
            results.append({
                    'score' : sim_score,
                    'id' : index,
                })
            # Sort results by score in asc order
            results.sort(key=lambda k : k['score'] , reverse=False)

        return results

