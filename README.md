# NEWS CATEGORY CLASSIFICATION USING RECURRENT NEURAL NETWORKS 
Comparing two different architectures of RNNs, the former a bidirectional GRU and the latter a bidirection LSTM, on how they train on a text categorization task.

---
## Technologies
- `Pytorch`
- `Natural Language ToolKit (NLTK)`
- `Scikit-learn`
- `Pandas, Numpy`
- `Netron.app`
- `Matplotlib`
---
## Model Architectures

### GRU (https://netron.app/?url=https://github.com/Asymmetric-OG/NewsClass/raw/refs/heads/master/grumodel.onnx)
<img width="5630" height="256" alt="grumodel onnx" src="https://github.com/user-attachments/assets/53d1bfe7-88bb-48b1-aa43-5e4196dc667c" />

### LSTM (https://netron.app/?url=https://github.com/Asymmetric-OG/NewsClass/raw/refs/heads/master/lstm.onnx)
<img width="5660" height="362" alt="lstm onnx" src="https://github.com/user-attachments/assets/eaf3e783-c219-4969-8b2a-311fa4f395a5" />

---
## Training-Validation Curves

<img width="1790" height="490" alt="graph" src="https://github.com/user-attachments/assets/2bd604fb-aef1-4d5f-b15c-4c40ed4a1336" />
Evidently, the GRU overfits early and heavily due to the vanishing gradients problems whereas the LSTM has more stable training due to its better performance on longer sequences of text.
Peak GRU validation accuracy : `71.5`
Peak LSTM validation accuracy : `66.2`

---
## File Overview

`Dataset.json` : News Category Classification Dataset.
`classifier.ipynb` : The entire workflow.
`grumodel.onnx` : Post-training GRU model for visualisation
`lstm.onnx` : Post-training GRU model for visualisation

---
