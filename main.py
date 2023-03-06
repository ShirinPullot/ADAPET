import torch
from transformers import AlbertForMaskedLM, AutoTokenizer
from torch.nn import functional as F
import uvicorn
from fastapi import FastAPI

from fastapi import FastAPI
model = AlbertForMaskedLM.from_pretrained('/Users/shirinwadood/Desktop/projects/Job_applications/Salezilaa/adapet_for_cloud/ADAPET/imdb_100_pattern1/', return_dict = True, is_decoder = True)
tokenizer = AutoTokenizer.from_pretrained('albert-xxlarge-v2')

app = FastAPI()


@app.get('/')
def get_root():
    return {'message': 'This is the sentiment analysis app'}


@app.get('/sentiment_analysis/')
def query_sentiment_analysis(text: str):
    return predict(text)


def get_patternized_text(sentiment):
    patternized_text= sentiment+ ' In summary, the review is [MASK]'
    # print(patternized_text)
    return patternized_text


def get_input_ids_for_tokenized_text(patternized_text):
    input_ids_of_patternizedtext = tokenizer.encode_plus(patternized_text, add_special_tokens=True, return_attention_mask=True, return_tensors="pt")
    # print('input ids are', input_ids_of_patternizedtext)
    return input_ids_of_patternizedtext

def predict(sentiment:str):
    pattern_text= get_patternized_text(sentiment)
    input_ids=get_input_ids_for_tokenized_text(pattern_text)
    output = model(**input_ids)
    positive_score = output.logits[0][-1][254]
    negative_score = output.logits[0][-1][896]
    print(positive_score, negative_score)
    if positive_score > negative_score:
        return 'review is positive'
    return 'review is negative'




