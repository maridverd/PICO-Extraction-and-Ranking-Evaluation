import re
from transformers import pipeline, AutoTokenizer


def sanitize_text(text: str) -> str:
    
    text =re.sub(r'^[^a-zA-Z0-9]+|(?<![a-zA-Z0-9]),|[^a-zA-Z0-9!?]+$', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text


class PicoClassifier():

    def __init__(self) -> None:
        MODEL = 'kamalkraj/BioELECTRA-PICO'
        tokenizer = AutoTokenizer.from_pretrained(MODEL, model_max_length=512, max_length=512)
        self.token_classifier = pipeline('token-classification', model=MODEL, tokenizer=tokenizer)


    def _extract_pico_from_text(self, text: str) -> dict:
        pico = {}

        results = self.token_classifier(text)
        for i, token in enumerate(results):
            if token['entity'] not in pico:
                pico[token['entity']] = []

            word = token['word'].replace('##', '')

            if i > 0 and results[i-1]['entity'] == token['entity']:
                dist = token['start'] - results[i-1]['end']

                if dist <= 1 or text[results[i-1]['end']:token['start']] == dist*' ':
                    pico[token['entity']][-1] += f"{dist * ' '}{word}"
                else:
                    pico[token['entity']].append(word)
            else:
                pico[token['entity']].append(word)

        for key, value in pico.items():
            sanitized_texts = []
            for v in value:
                if sanitized := sanitize_text(v):
                    sanitized_texts.append(sanitized)

                pico[key] = sanitized_texts

        return pico
    
    def extract_picos(self, abstracts: dict) -> dict:
        picos = {}
        for k, value in abstracts['data'].items():
            try:
                abstract = value['AbstractText']
                pico_result = {}

                if isinstance(abstract, list):
                    for ab in abstract:
                        pico_ab = self._extract_pico_from_text(ab['#text'])
                        for _class in set(list(pico_result.keys()) + list(pico_ab.keys())):
                            pico_result[_class] = pico_result.get(_class, []) + pico_ab.get(_class, [])
                else:
                    pico_result = self._extract_pico_from_text(abstract)

                # Se não encontrou nenhuma entidade, pula esse artigo e informa o erro
                if not pico_result:
                    raise Exception("Nenhuma entidade PICO encontrada no artigo")

                # Agora sim, salva no dicionário
                picos[k] = {
                    'picos': pico_result,
                    'ArticleTitle': value.get('ArticleTitle'),
                    'doi': value.get('doi')
                }

            except Exception as e:
                print(f'Error on article {k}: {e}')

        return picos

