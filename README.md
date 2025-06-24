# Sparse-Cost-Robust-Learning

Topic : Robust Instance-dependent reweighting method for Cost sensitive Learning with Sparse Costs.

The code of this repo proposes a new appraoch to integrate instance-cost information in the training phase of a neural network when the this cost information is only available for a very small minority of instances. 

![alt text](https://github.com/BruNicoletta/Sparse-Cost-Robust-Learning/blob/main/Images/AlgorithmSchematics.png) 

Some important links:
 - Link to the paper manuscript: [Paper - highly recommended read - Acces currently restricted: I will make it public as soon as paper is fully reviewed](https://fr.overleaf.com/project/675ce6fffe4b46458530152d) 
 - Link to special dedicated Google Collab:  [Series of Jupyter Notebooks showcasing proof of concepts and working principles - Acces currently restricted: I will make it public as soon as paper is fully reviewed](https://drive.google.com/drive/folders/1NC3Cig3cc60c_hBxwyBwHAVyQWuUoTOF?usp=sharing)
 

## Core Concept 

This repository introduces a novel approach to incorporating instance-dependent cost information into the training process of neural networks, specifically in scenarios where such cost data is available for only a small minority of the training instances.

In cost-sensitive learning, optimizing for expected real-world costs or profits requires loss functions that directly account for instance-level cost variability. However, in many practical settings, complete cost matrices are either partially observed or prohibitively expensive to obtain. Standard cost-sensitive training cannot be applied reliably when the cost information is sparse, and conventional imputation or sample discarding strategies often degrade model performance or introduce bias.

To address this, the method builds upon the meta-learning framework introduced by Ren et al. (2019) [HyperLink to be added properly]. Instead of learning weights to improve robustness to noisy labels, the method learns to reweight costless instances such that the resulting gradient updates improve performance on the subset of data with observed cost information. 

This reweighting strategy allows cost knowledge from instances with available cost infromation (used in inner loop) to guide the learning on samples without cost information (integrated in the outer loop). By doing so, the model can leverage the full training dataset while maintaining alignment with cost-sensitive objectives. The learned weights emphasize instances that exhibit patterns consistent with cost-aware decision-making and down-weight those that lead to gradient updates misaligned with the cost-sensitive objective.

**This approach offers a structured and scalable solution for training deep learning models when instance-level cost information is available for only a small subset of the data — a common limitation in real-world scenarios.**


## Algorithm overview

Algorithm pseudocode             |  Algorithm schematics
:-------------------------:|:-------------------------:
![](https://github.com/BruNicoletta/Sparse-Cost-Robust-Learning/blob/main/Images/Algorithm_pseudoCode.png)   |  ![](https://github.com/BruNicoletta/Sparse-Cost-Robust-Learning/blob/main/Images/AlgorithmSchematics.png) 


## Results

### UCI Bank Marketing: [Dataset - Kaggle link](https://archive.ics.uci.edu/dataset/222/bank+marketing)

 Example for 10% of cost data available    |  Results with varying sparisty levels
:-------------------------:|:-------------------------:
![alt text](https://github.com/BruNicoletta/Sparse-Cost-Robust-Learning/blob/main/Images/20%20June%20-%20preprocess_bank_marketing%20-%20many%20intervals/Comparison_val0.1_repeat1.png)   | ![alt text](https://github.com/BruNicoletta/Sparse-Cost-Robust-Learning/blob/main/Images/20%20June%20-%20preprocess_bank_marketing%20-%20many%20intervals/sparsity_levels(validationAvailable).png) 

### Default of Credit Card Clients: [Dataset - Kaggle link](https://archive.ics.uci.edu/dataset/350/default+of+credit+card+clients#)

 Example for 10% of cost data available    |  Results with varying sparisty levels
:-------------------------:|:-------------------------:
![alt text](https://github.com/BruNicoletta/Sparse-Cost-Robust-Learning/blob/main/Images/June%2017%20-%20default%20credit%20card%2013000%20with%20amount%20as%20feature/Comparison_val0.1_repeat1.png)   | ![alt text](https://github.com/BruNicoletta/Sparse-Cost-Robust-Learning/blob/main/Images/June%2017%20-%20default%20credit%20card%2013000%20with%20amount%20as%20feature/results_with_varying_sparsity_levels.png) 


## Absract of the paper

   
In many business applications, a predictive model that minimizes cost (or equivalently, maximizes profit) is strategically more valuable than one that optimizes accuracy. Instance-dependent cost-sensitive learning models are ideal for these tasks, as they generally exhibit superior cost-sensitive performance. 

However, obtaining instance-specific cost data can be challenging. What is the best approach when cost information is only available for a minority of observations? In this paper, we address the theoretical foundations underpinning this issue and develop a method to effectively utilize a sparse cost matrix. 

Our approach leverages the limited available cost information to optimize cost-efficient predictions. Specifically, our algorithm uses instances with known costs to assess the potential influence of instances lacking cost information on the overall cost, adjusting their weighting during model training accordingly. 

**This method achieves better performance on cost-sensitive metrics, at the expense of cost-insensitive metrics, and can be implemented for most classic neural networks**

**==> Link to the paper manuscript**:   
[Paper - highly recommended read - Acces currently restricted: I will make it public as soon as paper is fully reviewed](https://fr.overleaf.com/project/675ce6fffe4b46458530152d)

