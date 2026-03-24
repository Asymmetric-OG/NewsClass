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
GRU (https://netron.app/?url=https://github.com/Asymmetric-OG/NewsClass/raw/refs/heads/master/grumodel.onnx) 

<img width="5630" height="256" alt="grumodel onnx" src="https://github.com/user-attachments/assets/53d1bfe7-88bb-48b1-aa43-5e4196dc667c" />

LSTM (https://netron.app/?url=https://github.com/Asymmetric-OG/NewsClass/raw/refs/heads/master/lstm.onnx)  

<img width="5660" height="362" alt="lstm onnx" src="https://github.com/user-attachments/assets/eaf3e783-c219-4969-8b2a-311fa4f395a5" />

---
## Observations (Training-Validation Curves)

Evidently, the GRU overfits early and heavily due to the vanishing gradients problems whereas the LSTM has more stable training due to its better performance on 
longer sequences of text.

### LR=1e-3
<img width="1790" height="490" alt="graph" src="https://github.com/user-attachments/assets/2bd604fb-aef1-4d5f-b15c-4c40ed4a1336" />

Peak GRU validation accuracy : `66.2`  
Peak LSTM validation accuracy : `71.5` 

 ### LR=1e-5
 EPOCH(1-25)
<img width="1790" height="490" alt="one-twenty5" src="https://github.com/user-attachments/assets/23d81ddd-331c-47dc-b52d-27d9f83b7241" />

 EPOCH(25-50)
 <img width="1790" height="490" alt="twenty5-fifty" src="https://github.com/user-attachments/assets/93a335d7-3361-409c-bfab-328f9d88637e" /> 

Peak GRU validation accuracy : `60+`  (OVERFITTED)  
Peak LSTM validation accuracy : `60+` (GENERALISES WELL)  

 _This emphasizes on the LSTMs ability to tweak its gradients efficiently over a period of 50 epochs whereas they explode/vanish for the former model._
 
---
## File Overview
- `Dataset.json` : News Category Classification Dataset.
- `classifier.ipynb` : The entire workflow.
- `grumodel.onnx` : Post-training GRU model for visualisation
- `lstm.onnx` : Post-training GRU model for visualisation

---
