

class chat_plugin(Object):
    def __init__(self, tokenizer, model, prompt=''):
        self.tokenizer = tokenizer
        self.model = model
        self.prompt = prompt

    def generate(self, input_text):
        ids = self.tokenizer.encode(input_text)
        input_ids = torch.LongTensor([ids]).to(self.model.device)
        out = self.model.generate(
            input_ids=input_ids,
            max_new_tokens=400,
            do_sample=False,
            temperature=0
        )
        out_text = self.tokenizer.decode(out[0])


def call_search_plugin():



def call_kg_plugin():



