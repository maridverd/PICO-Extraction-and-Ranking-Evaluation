import re
import nltk
import torch
import numpy as np
import pandas as pd
from typing import List, Dict
from nltk.corpus import stopwords
from torch.nn import CosineSimilarity
from transformers import AutoTokenizer, AutoModel


nltk.download('stopwords')

ENGLISH_SW = stopwords.words('english')
#MODEL = 'dmis-lab/biobert-base-cased-v1.1'
MODEL = 'kamalkraj/BioSimCSE-BioLinkBERT-BASE' # Troquei para um modelo mais adept para sentence similarity 

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def clean_texts(texts: List[str]) -> List[str]:
    sanitized_texts = []
    
    if isinstance(texts, str):
        texts = [texts]

    for text in texts:
        text = text.lower()
        text = re.sub(r"(?<![a-z\d])[\.,\?\!](?![a-z\d])", "", text.replace("-", " "))
        text = re.sub(r"\s+", " ", text).strip()
        text = ' '.join([word for word in text.split() if word not in ENGLISH_SW])
        if len(text):
            sanitized_texts.append(text)

    return sanitized_texts

# Para utilizar mean pooling para geração do embedding invês do CLS token
def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    return torch.sum(token_embeddings * mask, 1) / torch.clamp(mask.sum(1), min=1e-9)


class Ranker():

    def __init__(self) -> None:
        self.cos = CosineSimilarity(dim=0, eps=1e-6)
        
        self.emb_tokenizer = AutoTokenizer.from_pretrained(MODEL)
        self.emb_text_encoder = AutoModel.from_pretrained(MODEL).to(DEVICE)


    def get_embeddings_from_text_array(self, text_array: List[str]) -> np.array:
        if not text_array or not len(text_array):
            return np.zeros((1, 768)) # Retorna vetor zerado se vazio para não quebrar
        
        text_inputs = self.emb_tokenizer(text_array, padding=True, truncation=True, max_length=512, return_tensors='pt').to(DEVICE)
        outputs = self.emb_text_encoder(**text_inputs)
        embeddings = mean_pooling(outputs, text_inputs['attention_mask']) # Usa mean pooling
       # embeddings = outputs.last_hidden_state[:, 0, :]  # Usa o [CLS] token, um token de classificação global de tamanho fixo (768)
        return embeddings.cpu().detach().numpy()


    def get_similarity_score_between_embeddings(self, embedding1: np.array, embedding2: np.array) -> float:
        similarity = self.cos(torch.from_numpy(embedding1), torch.from_numpy(embedding2))
        return similarity.cpu().detach().numpy()

    
    def get_mean_similarity_between_embeddings_and_texts(self, objs_embs: np.array, texts: List[str]) -> float:
        """Calcula a similaridade média entre um array de embeddings e uma lista de textos.

        Args:
            objs_embs (np.array): Embeddings dos valores chaves a serem pesquisados
            texts (List[str]): Lista de textos em que se deseja comparar a similaridade

        Returns:
            float: score médio utilizando similaridade de cosseno
        """
        texts = clean_texts(texts)
        if not texts: return 0.0 # Se não sobrou texto após limpeza, similaridade é 0
        texts_embs = self.get_embeddings_from_text_array(texts)
        num_combs = len(objs_embs)*len(texts_embs)

        scores = 0
        for obj_embs in objs_embs:
            for text_embs in texts_embs:
                scores += self.get_similarity_score_between_embeddings(obj_embs, text_embs)

        return scores/num_combs if num_combs > 0 else 0

    
    def rank_papers_by_similarity(self, objectives: Dict[str, List[str]], papers: Dict) -> pd.DataFrame:
        """Essa função retorna um dataframe ordenado com os papers de maior similaridade entre os
        picos.

        Args:
            objectives (Dict[str, List[str]]): Sentenças em que serão analisadas a similaridade
            papers (Dict): papers classificados por pico

        Returns:
            pd.DataFrame: Rank de papers.
        """
        ranks = []
        objs_embs = {} # Embeddings da query
        for k, v in objectives.items():
            cleaned = clean_texts(v)
            if cleaned:
                objs_embs[k] = self.get_embeddings_from_text_array(cleaned)

        for paper_id, paper_data in papers.items():

            paper_score = {
                'Id': paper_id,
                'ArticleTitle': paper_data['ArticleTitle'],
                'doi': paper_data['doi'],
                'Score': 0
            }

            for pico in ['I-Participants', 'I-Intervention', 'I-Outcome']:
                # Verifica se o PICO existe na QUERY (objs_embs) e no ARTIGO (papers)
                if pico in objs_embs and pico in paper_data['picos']:
                    score = self.get_mean_similarity_between_embeddings_and_texts(
                        objs_embs=objs_embs[pico],
                        texts=paper_data['picos'][pico]
                    )
                    paper_score[f'{pico[2:]}Score'] = score
                    paper_score['Score'] += score
                else:
                    paper_score[f'{pico[2:]}Score'] = 0

            paper_score['Score'] = paper_score['Score'] / 3
            ranks.append(paper_score)

        return pd.DataFrame(ranks).sort_values(by='Score', ascending=False)