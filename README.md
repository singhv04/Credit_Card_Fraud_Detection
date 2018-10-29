# Credit_Card_Fraud_Detection
This unsupervised system helps the bank to not only detect the frauds but also ranks the customers according to their probability of frauds.

Approach:
First used the self organizing map which takes a multi-dimensional dataset which might have dataset of very large number of dimensions.They reduce the dimensionality of dataset and we end up with a map which is a 2D reprsentation of dataset.Its learn to group them itself as its an unsupervised learning.They retain topology of the input itself and reveal co-relations that are not easily identified.They classify data without supervision so no targer vector needed.There need not be any lateral conections between output nodes.
So this model will identify some patterns in a high dimensional dataset full of non-linear dataset and one of these patterns will be potential fraud.
Once the pattern is found we will rank the potential frauds in order of the risk with the help of supervised model ANN.

I have attached the output of self organizing map as graph.png. 
