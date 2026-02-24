# Named Entity Recognition

## AIM

To develop an LSTM-based model for recognizing the named entities in the text.

## Problem Statement and Dataset
Named Entity Recognition (NER) is a sequence labeling task where the goal is to identify and classify named entities in text into predefined categories such as person, organization, location, date, and more. This project uses a BiLSTM model to learn contextual word representations and assign NER tags to each word in a sentence. The dataset used is a standard NER dataset containing three columns — Word, POS, and Tag — where each word is labeled with its corresponding BIO-style named entity tag across multiple sentences.

## DESIGN STEPS

### STEP 1: Data Loading and Preprocessing
The NER dataset is loaded and forward-filled to handle missing sentence identifiers. Words and tags are extracted and mapped to numerical indices. Sentences are grouped and encoded, then padded to a fixed length of 50 tokens. The data is split into 80% training and 20% test sets and wrapped in a custom PyTorch Dataset and DataLoader.

### STEP 2: Model Definition and Setup
A Bidirectional LSTM model is defined with an embedding layer, a BiLSTM layer, and a fully connected output layer that predicts a tag for each token. Cross-entropy loss is used as the criterion and Adam is used as the optimizer. The model is moved to GPU if available.

### STEP 3: Training and Evaluation
The model is trained for 3 epochs where each epoch computes training and validation loss. After training, the model is evaluated on the test set using a classification report. A sample sentence is passed through the model and predicted tags are compared against the actual tags word by word.


## PROGRAM
### Name: Ramitha Chowdary S
### Register Number: 212224240130
```python
class BiLSTMTagger(nn.Module):
    # Include your code here







    def forward(self, input_ids):
        # Include your code here
        


model = 
loss_fn = 
optimizer = 


# Training and Evaluation Functions
def train_model(model, train_loader, test_loader, loss_fn, optimizer, epochs=3):
    # Include the training and evaluation functions






    return train_losses, val_losses

```
## OUTPUT

### Training Loss, Validation Loss Vs Iteration Plot

Include your plot here

### Sample Text Prediction
Include your sample text prediction here.

## RESULT
