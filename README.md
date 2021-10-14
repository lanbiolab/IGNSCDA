# IGNSCDA
We propose a new computational model based on improved graph convolutional network and negative sampling to predict circRNA-disease associations. This is the implementation of
IGNSCDA:
```
Lan W, Dong Y, Chen Q, et al. IGNSCDA: Predicting CircRNA-Disease Associations Based on Improved Graph Convolutional Network and Negative Sampling[J]. IEEE/ACM Transactions on Computational Biology and Bioinformatics, 2021.

```

The code is inspired by Neural Graph Collabrorative Filtering, Xiang wang, Xiangnan He, Meng Wang et al. In SIGIR'19,PAris,France,July 21-25,2019.

# Environment Requirement
+ python == 3.6
+ tensorflow == 1.8.0
+ numpy == 1.14.3
+ scipy == 1.1.0
+ sklearn == 0.19.1
+ keras == 2.3.1
+ matplotlib == 3.1.1
+ theano == 1.0.4

# Dataset
+ circRNA-disease-foldx: IGNSCDA is verified in 5-fold cross validation. The pair of circRNA-disease associations are split as test.txt and train.txt in corresponding fold files.
+ circRNA_disease_gcn_embedding_32_feature_file: The embeddings of circRNAs and diseases are recorded in this file. There are 5 files represent the embeddings of circRNAs and diseases in corresponding fold.
+ one_list_file: one_list.h5 records the pair of circRNA-disease associations. train_test_foldx.h5 saves the associations between circRNAs and diseases in each fold.


You can extract the data in one_list.h5 as:
```
with h5py.File('./one_list_file/one_list') as hf:
   one_list = f['one_list'][:]
```
+ disease-circRNA.h5: The circRNA-disease association matrix is recorded in disease-circRNA.h5, the matrix size is 533 (circRNA) * 99 (disease). The extracting method is as:
```
with h5py.File('./disease-circRNA.h5', 'r') as f:
    circrna_disease_matrix = f['infor'][:]
```

# Model
+ ExtractFeature.py: After propagating in IGCN, the embeddings of circRNAs and diseases will be concatenated and predicted in MLP. The model of mlp and the process of constructing train data and test data are recorded in this file.
+ IGNSCDA.py: The core method of IGNSCDA.
+ MakeSimilarityMatrix.py: This file records the detail of computing Gaussian similarities.
+  parser.py: The paramaters of IGNSCDA are adjusted in this file.
+  sortscore.py: The prediction score of each circRNA-disease pair is sorted in this file before computing the AUC/AUPR.

# Questions
+ If you have any questions or find the problems in this paper or code, please contact with me: Yi Dong: dongyi@st.gxu.edu.cn
