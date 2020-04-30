class MultiTaskClassifier(nn.Module):
    def __init__(self, hidden_dim, num_labels):
        super(MultiTaskClassifier, self).__init__()
        self.hidden_dim = hidden_dim
        self.num_labels = num_labels
        
        self.bertmodel = BertModel.from_pretrained('bert-base-uncased')
        self.ffn1 = nn.Linear(768, hidden_dim)
        self.dp1 = nn.Dropout()
        self.ffn2 = nn.Linear(hidden_dim, num_labels)
        
    def forward(self, in_T, in_T_attn_masks):
        hidden_states, _ = self.bertmodel(in_T, in_T_attn_masks)
        x = torch.mean(hidden_states, dim=1)
        x = F.relu(self.ffn1(x))
        x = self.dp1(x)
        x = torch.sigmoid(self.ffn2(x))
        return x