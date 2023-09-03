import pickle

sentiment_dict = {}
with open(f"evo.pkl","rb") as f:
  sentiment_dict = pickle.load(f)

for key,value in sentiment_dict.items():
  print(f"KEY: {key}")
  print(f"VALUE: {value}")
  print()