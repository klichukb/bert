# Toxic-Comment-Classifier-BERT
BERT Model for Toxic Comment Classification Dataset from Kaggle.

Toxic Comments are huge headache on the social platforms. Hence automatic detection and filtering of such content is a necessary task.

In this project I define a toxic comment classifier incorporating the state-of-the-art BERT model for classification using Pytorch library.

## Steps to run the model

```
git clone https://github.com/dhanushsr/Toxic-Comment-Classifier-BERT.git
cd Toxic-Comment-Classifier-BERT
pip install -r requirements.txt
python main.py
```

### Project Directory
 
`data/dataset.csv` - Dataset used to train the model.  
`dataloader.py` - Contains the function used to load the dataset into the model.    
`main.py` - Main python file used to run the model.  
`model.py` - Contains the proposed model definition.
`model_state_dict.pt` - This is not included in git. This is the weights of the trained model. This can be downloaded [here](https://drive.google.com/open?id=1nftoJ6zOPt3OcfGU3fHtfl5WQXKbhT4t).  
`toxic_comment_classifier_result.txt` - Output of the instance run of the model.  
`training_stats.csv` - Training Stats of the run of the model.  
`training_stats_pickle` - Training Stats of the run of the model.
`utils.py` - Contains the utility functions used in main.py.

## Dataset Used:

URL : <https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge>  
No of samples: 159571  
Labels: `toxic, severe_toxic, obscene, threat, insult, identity_hate`  

### Results:
Overall Accuracy : `0.9841`  
Overall Macro-F1 score: `0.8328`  
Overall Weighted-F!-score: `0.9843`  
