import torch
from model import MultiTaskClassifier
from transformers import BertTokenizer, BertModel, AdamW, get_linear_schedule_with_warmup

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased', do_lower_case=True)
bertmodel = BertModel.from_pretrained('bert-base-uncased')


device = torch.device('cpu')
state = torch.load('model_state_final.pt', map_location=torch.device('cpu'))
model = MultiTaskClassifier(200, 6).to(device)
model.load_state_dict(state)


def predict(sentences):
    max_length = 512
    enc_sent_dict = tokenizer(
        [sentence[:max_length] for sentence in sentences],
        max_length = max_length,
        add_special_tokens = True,
        padding='max_length',
        return_attention_mask = True,
        return_tensors = 'pt'
    )
    in_T = enc_sent_dict['input_ids']
    in_T_attn_masks = enc_sent_dict['attention_mask']
    res = model(in_T, in_T_attn_masks)
    # res = bertmodel(in_T, in_T_attn_masks)
    return res



def realign(h_counts, b, mmin, mmax):
    approx_data = []
    for count, (start, end) in zip(h_counts, zip(b[:-1], b[1:])):
        approx_data.extend(np.linspace(start, end, int(count)))
    new_bin_edges = np.linspace(mmin, mmax, len(b))
    new_bin_counts, _ = np.histogram(approx_data, bins=new_bin_edges)
    return new_bin_counts, new_bin_edges

import pandas as pd

df = pd.read_csv('data/dataset.csv')
baseline_df = df.sample(len(df) // 10, random_state=42)
# baseline_df = df.sample(3, random_state=42)
base = baseline_df.comment_text

result = predict(base)
import pdb; pdb.set_trace()

embeddings = model.res.last_hidden_state
torch.save(embeddings, "embeddings.pt")
baseline_df.to_csv("baseline.csv", index=False)
