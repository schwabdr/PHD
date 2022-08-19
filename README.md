# PHD
Final (hopefully) repository for my code for PHD - Evaluating MIAT

My evaluation of "Improving Adversarial Robustness via Mutual Information Estimation" - http://arxiv.org/abs/2207.12203. 

Also see the official code base for the paper: https://github.com/dwDavidxd/MIAT

My Goals: 
<ol>
<li>Provide a thorough evaluation of the Mutual Information Adversarial Training (MIAT) defensive scheme</li>
<li>Use the Mutual Information (MI) estimation networks as a metric to aid in the crafting of adversarial examples</li>
<li>Will the adversarial examples crafted from step 2 “break” the MIAT defense?
<ol>
<li>Evaluate these new adversarial examples against a MIAT trained target model.</li>
<li>Evaluate these new adversarial examples against a non-robust model and compare to PGD crafted examples.</li>
</ol>
</li>
<li>Can the adversarial examples crafted from step 2 be used again for adversarial training to make the network more robust?</li>
<li>Can we craft adversarial examples that were crafted from the estimator networks with the goal of reducing the effectiveness of the AT for the target network</li>
</ol>

Notes For running this project (primarily for myself right now). 
<ol>
<li>Download the CIFAR dataset and put in data folder (decompress file with terminal). (http://www.cs.toronto.edu/~kriz/cifar.html) </li>
<li>Run setup_CIFAR10.py to manipulate raw data into numpy arrays on disk.</li>
<li>Run train_standard_C10.py, make note of the name you give for the trained model. This will train a ResNet18 on CIFAR10 clean samples.</li>
<li>Run create_adv_examples.py. This script will create multiple datasets of adversarial examples. It will keep correct labels for each image. np array format again.</li>
<li>Run train_at_C10.py, this file will create a new ResNet18 model, adversarially trained using standard AT (not the MIAT method). Again make a note of the name you give this model, you will need it later.</li>
</ol>
