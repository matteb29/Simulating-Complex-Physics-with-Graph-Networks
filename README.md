# Simulating-Complex-Physics-with-Graph-Networks

This is our Project for the CMEPDA exam.

It consists of a simplified version of the Graph-Net Simulator introduced by *Sanchez-Gonzales et al.* in their paper **"Learning to Simulate Complex Physics with Graph Networks"(ICML 2020)**.

We reduced the problem to simulate the temporal evolution
of a complex system inside a 2D box (without obstacles in).
As another simplification we considered only two separate materials: Water and Sand.


<p align="center">
  <img src="assets/simulation_demo.gif" width="800" />
  <br>
  <em> Example Result: Comparison between Ground Truth (Left) and GN Prediction (Right) for Sand.</em>
</p>


# How to use the Simulator:

`Datasets` are all available at the following repository: **https://github.com/google-deepmind/deepmind-research/tree/master/learning_to_simulate**

*Note* that, in order to run our code, data must be *converted from .tfrecord to .npz*
and must preserve the original structure: train(directory), test(directory), valid(directory), metadata(JSON file).


Clone the repository:
```bash
git clone https://github.com/matteb29/Simulating-Complex-Physics-with-Graph-Networks.git
```
Move into the directory:
```bash
cd Simulating-Complex-Physics-with-Graph-Networks
```

`NOTE` that in this project we use both torch-cluster and torch-scatter.
These must be installed matching your Pytorch and CUDA versions, see the 
official installation guide https://pytorch-geometric.readthedocs.io/en/latest/install/installation.html for instructions. Then download the other requirements:
```bash
pip install -r requirements.txt
```


Train the model with a specific Dataset (e.g. "Sand" in "datasets" directory):
```bash
mkdir model_checkpoints
python train.py --mode=train --data_path=datasets/Sand --model_path=model_checkpoints
```


Create a rollout with the test set using a specific model (e.g. "model_1000.pt")
```bash
mkdir rollout
python train.py --mode=eval_rollout --eval_split=test --data_path=datasets/Sand --model_path=model_checkpoints/model_1000.pt --output_path=rollout
```

Visualize the comparison between Ground Truth and the generated rollout as a GIF (e.g. "rollout_0.pkl")
```bash
mkdir video_simulation
python render_rollout.py --rollout_path=rollout/rollout_0.pkl --output_path=video_simulation/rollout_0.gif
```

# Code Structure
In this repository you can find the following scripts:

`graph_net.py`: implements the network framework structured as **Encoder-Processor-Decoder**

`learned_simulator.py`: implementation of a Learnable Simulator based on the Encoder-Processor-Decoder
architecture

`train.py`: this script implements the pipeling of both training and evaluation of our Simulator

`reading_utils.py`: a module to split the dataset in simulation's "frame windows" 

`noise_utils.py`: a module to inject noise during training 

`render_rollout.py`: a module to visualize the rollout results as a GIF

**Note** that the we ran the scripts using CERN SWAN environment so they are optimized to it. Every single training was done using a Tesla T4 GPU and last about 20 hours.







