import torch
from transformers import AutoModel, AutoTokenizer
from peft import PeftModel

class FinGpt:
    def __init__(self):
        self.device = "cuda:0" if torch.cuda.is_available() else "cpu"
        print(f'Initializing FinGPT with: {self.device}')
        torch.cuda.empty_cache()

        self.tokenizer, self.model = self.__load_model_tokenizer()

    def __load_model_tokenizer(self):
        base_model = "THUDM/chatglm2-6b"
        peft_model = "oliverwang15/FinGPT_ChatGLM2_Sentiment_Instruction_LoRA_FT"
        tokenizer = AutoTokenizer.from_pretrained(
            base_model, trust_remote_code=True)
        model = AutoModel.from_pretrained(
            base_model, trust_remote_code=True,  device_map="auto")
        model = PeftModel.from_pretrained(model, peft_model)
        #model = model.to(self.device)
        model = model.eval()
        return tokenizer, model

    def interact(self, input: str) -> str:
      if not self.__is_valid_and_not_null(input):
        raise "Input is not valid"
      
      prompt = input

      # Generate results
      tokens = self.tokenizer(prompt, return_tensors='pt',
                              padding='max_length', max_length=500)

      tokens = tokens.to(self.device)
      res = self.model.generate(**tokens, max_length=512)
      res_sentences = [self.tokenizer.decode(i) for i in res]

      return res_sentences[0]
    
    def __is_valid_and_not_null(self, input_string: str) -> bool:
      if input_string is not None and isinstance(input_string, str) and len(input_string) > 0:
        return True
      return False

