from collections import Counter
import numpy as np
import math
import torch
import torch.nn as nn
import random
import time
import io
import codecs
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from numpy import nan, inf
import networkx as nx


from utils import evaluate_rouge
from bert_model import BertEdgeScorer, BertConfig

class PacSumExtractor:

    def __init__(self, extract_num = 3, beta = 3, lambda1 = -0.2, lambda2 = -0.2):

        self.extract_num = extract_num
        self.beta = beta
        self.lambda1 = lambda1
        self.lambda2 = lambda2

    def extract_summary(self, data_iterator):

        summaries = []
        references = []

        for item in data_iterator:
            article, abstract, inputs = item
            if len(article) <= self.extract_num:
                summaries.append(article)
                references.append([abstract])
                continue

            edge_scores = self._calculate_similarity_matrix(*inputs)
            ids = self._select_tops(edge_scores, beta=self.beta, lambda1=self.lambda1, lambda2=self.lambda2)
            summary = list(map(lambda x: article[x], ids))
            summaries.append(summary)
            references.append([abstract])

        result = evaluate_rouge(summaries, references, remove_temp=True)

    def tune_hparams(self, data_iterator, example_num=1000):


        summaries, references = [], []
        k = 0
        for item in data_iterator:
            article, abstract, inputs = item
            edge_scores = self._calculate_similarity_matrix(*inputs)
            tops_list, hparam_list = self._tune_extractor(edge_scores)

            summary_list = [list(map(lambda x: article[x], ids)) for ids in tops_list]
            summaries.append(summary_list)
            references.append([abstract])
            k += 1
            print(k)
            if k % example_num == 0:
                break

        best_rouge = 0
        best_hparam = None
        for i in range(len(summaries[0])):
            print("threshold :  "+str(hparam_list[i])+'\n')
            #print("non-lead ratio : "+str(ratios[i])+'\n')
            result = evaluate_rouge([summaries[k][i] for k in range(len(summaries))], references, remove_temp=True, rouge_args=[])

            if result['rouge_1_f_score'] > best_rouge:
                best_rouge = result['rouge_1_f_score']
                best_hparam = hparam_list[i]

        print("The best hyper-parameter :  beta %.4f , lambda1 %.4f, lambda2 %.4f " % (best_hparam[0], best_hparam[1], best_hparam[2]))
        print("The best rouge_1_f_score :  %.4f " % best_rouge)

        self.beta = best_hparam[0]
        self.lambda1 = best_hparam[1]
        self.lambda2 = best_hparam[2]

    def _calculate_similarity_matrix(self, *inputs):

        raise NotImplementedError


    def _select_tops(self, edge_scores, beta, lambda1, lambda2):

        min_score = edge_scores.min()
        max_score = edge_scores.max()
        edge_threshold = min_score + beta * (max_score - min_score)
        new_edge_scores = edge_scores - edge_threshold
        forward_scores, backward_scores, _ = self._compute_scores(new_edge_scores, 0)
        forward_scores = 0 - forward_scores

        paired_scores = []
        for node in range(len(forward_scores)):
            paired_scores.append([node,  lambda1 * forward_scores[node] + lambda2 * backward_scores[node]])

        #shuffle to avoid any possible bias
        random.shuffle(paired_scores)
        paired_scores.sort(key = lambda x: x[1], reverse = True)
        extracted = [item[0] for item in paired_scores[:self.extract_num]]


        return extracted

    def _compute_scores(self, similarity_matrix, edge_threshold):

        forward_scores = [0 for i in range(len(similarity_matrix))]
        backward_scores = [0 for i in range(len(similarity_matrix))]
        edges = []
        for i in range(len(similarity_matrix)):
            for j in range(i+1, len(similarity_matrix[i])):
                edge_score = similarity_matrix[i][j]
                if edge_score > edge_threshold:
                    forward_scores[j] += edge_score
                    backward_scores[i] += edge_score
                    edges.append((i,j,edge_score))

        return np.asarray(forward_scores), np.asarray(backward_scores), edges


    def _tune_extractor(self, edge_scores):

        tops_list = []
        hparam_list = []
        num = 10
        for k in range(num + 1):
            beta = k / num
            for i in range(11):
                lambda1 = i/10
                lambda2 = 1 - lambda1
                extracted = self._select_tops(edge_scores, beta=beta, lambda1=lambda1, lambda2=lambda2)

                tops_list.append(extracted)
                hparam_list.append((beta, lambda1, lambda2))

        return tops_list, hparam_list


class PacSumExtractorWithBert(PacSumExtractor):

    def __init__(self, bert_model_file, bert_config_file, extract_num = 3, beta = 3, lambda1 = -0.2, lambda2 = -0.2):

        super(PacSumExtractorWithBert, self).__init__(extract_num, beta, lambda1, lambda2)
        self.model = self._load_edge_model(bert_model_file, bert_config_file)

    def _calculate_similarity_matrix(self,  x, t, w, x_c, t_c, w_c, pair_indice):
        #doc: a list of sequences, each sequence is a list of words

        def pairdown(scores, pair_indice, length):
            #1 for self score
            out_matrix = np.ones((length, length))
            for pair in pair_indice:
                out_matrix[pair[0][0]][pair[0][1]] = scores[pair[1]]
                out_matrix[pair[0][1]][pair[0][0]] = scores[pair[1]]

            return out_matrix

        scores = self._generate_score(x, t, w, x_c, t_c, w_c)
        doc_len = int(math.sqrt(len(x)*2)) + 1
        similarity_matrix = pairdown(scores, pair_indice, doc_len)

        return similarity_matrix

    def _generate_score(self, x, t, w, x_c, t_c, w_c):

        #score =  log PMI -log k
        scores = torch.zeros(len(x)).cuda()
        step = 20
        for i in range(0,len(x),step):

            batch_x = x[i:i+step]
            batch_t = t[i:i+step]
            batch_w = w[i:i+step]
            batch_x_c = x_c[i:i+step]
            batch_t_c = t_c[i:i+step]
            batch_w_c = w_c[i:i+step]

            inputs = tuple(t.to('cuda') for t in (batch_x, batch_t, batch_w, batch_x_c, batch_t_c, batch_w_c))
            batch_scores, batch_pros = self.model(*inputs)
            scores[i:i+step] = batch_scores.detach()


        return scores

    def _load_edge_model(self, bert_model_file, bert_config_file):

        bert_config = BertConfig.from_json_file(bert_config_file)
        model = BertEdgeScorer(bert_config)
        model_states = torch.load(bert_model_file)
        print(model_states.keys())
        model.bert.load_state_dict(model_states)

        model.cuda()
        model.eval()
        return model


class PacSumExtractorWithTfIdf:
    def __init__(self, extract_num=3, beta=0, lambda1=0, lambda2=1):
        """Initialize PacSum extractor with TF-IDF"""
        self.extract_num = extract_num
        self.beta = beta
        self.lambda1 = lambda1 
        self.lambda2 = lambda2
        self.vectorizer = TfidfVectorizer(stop_words='english')
        
    def build_similarity_matrix(self, sentences):
        """Build similarity matrix between sentences using TF-IDF"""
        # Create TF-IDF matrix
        tfidf_matrix = self.vectorizer.fit_transform(sentences)
        
        # Calculate cosine similarity
        similarity_matrix = (tfidf_matrix * tfidf_matrix.T).toarray()
        
        # Normalize similarity scores to [0,1]
        similarity_matrix = np.nan_to_num(similarity_matrix)
        np.fill_diagonal(similarity_matrix, 0)
        
        return similarity_matrix
        
    def calculate_scores(self, similarity_matrix):
        """Calculate importance scores using PacSum algorithm"""
        num_sentences = similarity_matrix.shape[0]
        
        # Forward and backward scores
        forward_scores = np.zeros(num_sentences)
        backward_scores = np.zeros(num_sentences)
        
        # Calculate scores for each sentence
        for i in range(num_sentences):
            # Forward score - similarity with following sentences
            if i < num_sentences - 1:
                forward_scores[i] = np.mean(similarity_matrix[i, i+1:])
            
            # Backward score - similarity with preceding sentences  
            if i > 0:
                backward_scores[i] = np.mean(similarity_matrix[i, :i])
                
        # Combined score using PacSum formula
        final_scores = self.lambda1 * forward_scores - self.lambda2 * backward_scores + self.beta
        
        return final_scores
    
    def summarize(self, sentences, num_sentences=None):
        """Generate summary by extracting most important sentences"""
        if not sentences:
            return []
            
        if num_sentences is None:
            num_sentences = self.extract_num
            
        # Build similarity matrix
        similarity_matrix = self.build_similarity_matrix(sentences)
        
        # Calculate sentence scores
        sentence_scores = self.calculate_scores(similarity_matrix)
        
        # Get indices of top scoring sentences while preserving order
        top_indices = np.argsort(sentence_scores)[-num_sentences:]
        top_indices.sort()
        
        # Extract selected sentences
        summary = [sentences[i] for i in top_indices]
        
        return summary

class PacSumExtractorWithBert:
    def __init__(self, bert_model_file, bert_config_file, beta=0, lambda1=0, lambda2=1):
        # Placeholder for BERT implementation
        pass

    def tune_hparams(self, dataset_iterator):
        pass

    def extract_summary(self, dataset_iterator):
        pass
