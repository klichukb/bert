def get_dataset(df, tokenizer, mode='train'):
    sentences, labels = df['comment_text'], df.iloc[:,2:].to_numpy()
    max_length = 300
    in_T = []
    in_T_attn_masks = []
    for sentence in sentences:
        enc_sent_dict = tokenizer.encode_plus(
            sentence[:300],
            max_lenght = max_length,
            add_special_tokens = True,
            pad_to_max_length = True,
            return_attention_mask = True,
            return_tensors = 'pt'
        )
        in_T.append(enc_sent_dict['input_ids'])
        in_T_attn_masks.append(enc_sent_dict['attention_mask'])
    
    in_T = torch.cat(in_T, dim=0)
    in_T_attn_masks = torch.cat(in_T_attn_masks, dim=0)
    labels = torch.tensor(labels, dtype = torch.float32)
    print('Text Input: ' , in_T.shape)
    print('Text Input Attention: ' , in_T_attn_masks.shape)    
    print('Labels: ' , labels.shape)
    
    dataset = TensorDataset(
        in_T,
        in_T_attn_masks,
        labels
    )
    
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])
    return train_dataset, val_dataset