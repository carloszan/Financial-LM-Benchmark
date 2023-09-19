import pandas as pd
from tqdm import tqdm
import pickle
from ..models import FinGpt

finGpt = FinGpt()
print()

df = pd.read_csv("out.csv",index_col= 0)

sentences = df.text.unique()

bar = tqdm(total=len(sentences))

res_dict = {}

def get_gpt_res(sentence):
    sentence = f"Decide whether a sentence's sentiment is positive, neutral, or negative.\n\nSentence: \"{sentence}\"\nSentiment: "

    response = finGpt.interact(sentence)

    return response

def save_dict(dic):
    with open(f"res/missing.pkl","wb") as f:
        pickle.dump(dic,f)


evo = 0
for sentence in sentences:
    bar.update()

    res = get_gpt_res(sentence)
    res_dict[sentence] = res
    
    if evo % 1000 == 0:
        save_dict(res_dict)

    evo += 1


save_dict(res_dict)