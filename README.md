# PHD
Final (hopefully) repository for my code for PHD - Evaluating MIAT

My evaluation of "Improving Adversarial Robustness via Mutual Information Estimation" - http://arxiv.org/abs/2207.12203. 

Also see the official code base for the paper: https://github.com/dwDavidxd/MIAT

My Goals: 
1. Provide a thorough evaluation of the Mutual Information Adversarial Training (MIAT) defensive scheme
2. Use the Mutual Information (MI) estimation networks as a metric to aid in the crafting of adversarial examples
3. Will the adversarial examples crafted from step 2 “break” the MIAT defense?
  a. Evaluate these new adversarial examples against a MIAT trained target model.
  b. Evaluate these new adversarial examples against a non-robust model and compare to PGD crafted examples.
4. Can the adversarial examples crafted from step 2 be used again for adversarial training to make the network more robust?
5. Can we craft adversarial examples that were crafted from the estimator networks with the goal of reducing the effectiveness of the AT for the target network

