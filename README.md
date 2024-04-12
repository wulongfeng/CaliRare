# CaliRare

This is the implementation of our paper: [Towards Reliable Rare Category Analysis on Graphs via Individual Calibration](https://dl.acm.org/doi/abs/10.1145/3580305.3599525).


## Introduction

Rare categories abound in a number of real-world networks and play a pivotal role in a variety of high-stakes applications, including financial fraud detection, network intrusion detection, and rare disease diagnosis. Rare category analysis (RCA) refers to the task of detecting, characterizing, and comprehending the behaviors of minority classes in a highly-imbalanced data distribution. 
While the vast majority of existing work on RCA has focused on improving the prediction performance, a few fundamental research questions heretofore have received little attention and are less explored: \emph{How confident or uncertain is a prediction model in rare category analysis? How can we quantify the uncertainty in the learning process and enable reliable rare category analysis? }

To answer these questions, we start by investigating miscalibration in existing RCA methods. Empirical results reveal that state-of-the-art RCA methods are mainly under-confident in predicting minority classes and over-confident in predicting majority classes. Motivated by the observation, we propose a novel individual calibration framework, named \name, for alleviating the unique challenges of RCA, thus enabling reliable rare category analysis. In particular, to quantify the uncertainties in RCA, we develop a node-level uncertainty quantification algorithm to model the overlapping support regions with high uncertainty; to handle the rarity of minority classes in miscalibration calculation, we generalize the distribution-based calibration metric to the instance level and propose the first individual calibration measurement on graphs named Expected Individual Calibration Error (EICE). We perform extensive experimental evaluations on real-world datasets, including rare category characterization and model calibration tasks, which demonstrate the significance of our proposed framework. 



## Example to run CaliRare

	./example.sh
  
