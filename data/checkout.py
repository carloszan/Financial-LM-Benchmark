import pickle

sentiment_dict = {}
with open(f"text-curie-001.pkl","rb") as f:
  sentiment_dict = pickle.load(f)

#print(sentiment_dict.keys())
for key,value in sentiment_dict.items():
  print(f"KEY: {key}")
  print(f"VALUE: {value}")
  print()
  break