

def generate(input_text, model, tokenizer):
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



def call_schedule():


def call_chat_with_user(query, model, tokenizer):
    out = generate(query, model, tokenizer)[len(query):]








