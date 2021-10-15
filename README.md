# Network Science Assignment 3

## Overview
In this assignment, we first compare the performance of node classification algorithms on two classes of datasets, the `real-classic` datasets comprising mainly social networks from various sources and `real-label` datasets which are three citation networks taken from the repository of [1]. We used the two node classification algorithms from [networkx](https://networkx.org/documentation/stable/reference/algorithms/node_classification.html) and a crude logistic regression stacking model that tries to learn from the predictions made by the oher two.

In the second part, we compare the performance of various [link prediction](https://networkx.org/documentation/stable/reference/algorithms/link_prediction.html) algorithms. Once again, we fit logitic regression stacking models that attempt to use the predictions made by the various library algorithms to enhance the predictions, with mixed results.

---
<br />

## Setup
- unzip data.zip into data/
  
- create a virtual environment and activate it 
  ```
  python -m venv .venv 
  .venv/Scripts/activate
  ```
- download the requirements 
    ```
    pip install -r requirements.txt
    ```
- run the code:
    ```
    python ./A3.py
    ``` 
---
<br />

## References

[1] Thomas N. Kipf, Max Welling, Semi-Supervised Classification with Graph Convolutional Networks (ICLR 2017) https://github.com/tkipf/gcn/tree/39a4089fe72ad9f055ed6fdb9746abdcfebc4d81