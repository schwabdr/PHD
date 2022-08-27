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
<li>(Not complete yet) Run train_at_C10.py, this file will create a new ResNet18 model, adversarially trained using standard AT (not the MIAT method). Again make a note of the name you give this model, you will need it later.</li>
<li>Run make_adv_examples.py. This will create a dataset of adverarial examples from your ResNet18 model (depending on your GPUs can adjust batch size to get better utilization)</li>
<li>Run train_MI_estimator.py. This will take a while.</li>
<li>Finally, run train_MIAT_alpha.py. This will take a really long time. I'm going to work on speeding this up at some point </li>
<li>To run adversarial attacks with varying L_infty norm constraints, run eval_tests_01.py
</ol>

System specs for my work. Special thanks to Tommy Gorham and his project for this data: https://github.com/tommygorham/unv-smi

<pre>
##### Your Current System Configuration and Computational Resources Available #####
OS name: "Ubuntu 20.04.4 LTS"
CPU Architecture: x86_64
CPU Cores per Socket: 12
CPU Logical Cores: 24
0-23
CPU Name: Intel(R) Core(TM) i9-9920X CPU @ 3.50GHz
CPU Sockets Installed:  1
CPU Threads Per Core: 2
CacheLine Size: LEVEL1_ICACHE_LINESIZE 64
LEVEL1_DCACHE_LINESIZE 64
LEVEL2_CACHE_LINESIZE 64
LEVEL3_CACHE_LINESIZE 64
LEVEL4_CACHE_LINESIZE 0
Stack Size Limit: stack(kbytes) 8192
CPU Total Physical Cores: 12
GPU(s) detected:
19:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A] (rev a1)
1a:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A] (rev a1)
67:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A] (rev a1)
68:00.0 VGA compatible controller: NVIDIA Corporation TU102 [GeForce RTX 2080 Ti Rev. A] (rev a1)


##### Parallel Programming Environment #####
C++ Standard: C++17, 201703L
OpenMP Version: 4.5
GPU Programming Model: CUDA is the standard programming model for NVIDIA accelerators


##### Further Commands that can potentially be used for GPU identification #####
lspci | grep 3D
lspci |grepVGA
sudo lshw -C video
____________________________________________________________________________________

Thank you for using Universal System Management Interface version 1.0
</pre>
