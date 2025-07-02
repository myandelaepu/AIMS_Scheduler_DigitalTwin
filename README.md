# AIMS: Adaptive Intelligent Multi-objective Scheduling System
# https://colab.research.google.com/github/myandelaepu/AIMS_Scheduler_DigitalTwin/blob/main/AIMS_Scheduler_DigitalTwin.ipynb
# Overview
# AIMS (Adaptive Intelligent Multi-objective Scheduling System) is an advanced HPC (High Performance Computing) scheduler that integrates Digital Twin technology with Deep Reinforcement Learning for multi-objective optimization. The system optimizes three key objectives:

1.) Energy Efficiency: Minimizing power consumption and carbon footprint
2.) Performance: Maximizing computational throughput and resource utilization
3.) Reliability: Reducing system failures and ensuring fault tolerance

# Features
# Digital Twin Integration

1.) Predictive Fault Model: Anticipates system failures before they occur
2.) Energy Prediction Model: Forecasts power consumption patterns
3.) Performance Prediction Model: Estimates job execution characteristics
4.) Thermal Model: Monitors and predicts system temperature dynamics

#  Advanced AI Components

1.) Dueling DQN: State-of-the-art deep reinforcement learning architecture
2.) Multi-objective Optimization: Simultaneous optimization of competing objectives
3.) Uncertainty-aware Decision Making: Incorporates prediction uncertainty into scheduling decisions
4.) Memory-efficient Implementation: Optimized for Google Colab and resource-constrained environments

# Real-world Dataset Support

1.) Compatible with ANL-ALCF HPC datasets
2.) Supports multiple data formats (CSV, compressed files)
3.) Automatic data preprocessing and feature engineering
4.) Synthetic data generation for testing when real data is unavailable

# Dataset Information
# This project uses the ANL-ALCF HPC Workload Dataset available from IEEE DataPort:

# https://ieee-dataport.org/documents/argonne-leadership-computing-facility-data-catalog
# Description: Real-world HPC workload traces from Argonne National Laboratory's Leadership Computing Facility
# Systems Included: Aurora, Polaris, Mira, Cooley supercomputers
# Time Period: 2019-2025 (varies by system)

# Quick Start
# Option 1: Google Colab (Recommended)
# Click the "Open in Colab" badge above or use this direct link:
# AIMS_Scheduler_DigitalTwin.ipynb - Colab
# Option 2: Local Installation
# Prerequisites

# Python 3.7+
# PyTorch 1.9+
# CUDA (optional, for GPU acceleration)

# Installation Steps
1.) Clone the repository:
# git clone https://github.com/myandelaepu/AIMS_Scheduler_DigitalTwin.git
# cd AIMS_Scheduler_DigitalTwin
2.) Install dependencies:
# pip install torch torchvision numpy pandas scikit-learn matplotlib seaborn

3.) Download the dataset (optional):
a) Visit IEEE DataPort or https://reports.alcf.anl.gov/data/index.html
b) Download the ANL-ALCF dataset files
c) Place them in the project directory

4.) Run the system:
# python AIMS_Scheduler_DigitalTwin.py
# System Architecture

┌─────────────────────────────────────────────────────────────┐
│                    AIMS Architecture                        │
├─────────────────────────────────────────────────────────────┤
│  ┌─────────────────┐    ┌─────────────────┐                │
│  │   Data Loader   │    │ Digital Twin    │                │
│  │  - HPC Datasets │    │ - Fault Model   │                │
│  │  - Preprocessing│    │ - Energy Model  │                │
│  │  - Feature Eng. │    │ - Perf Model    │                │
│  └─────────────────┘    │ - Thermal Model │                │
│           │              └─────────────────┘                │
│           ▼                       │                         │
│  ┌─────────────────┐              ▼                         │
│  │ HPC Environment │    ┌─────────────────┐                │
│  │ - Job Queue     │◄───┤  Dueling DQN    │                │
│  │ - System State  │    │ - Feature Ext.  │                │
│  │ - Multi-obj     │    │ - Value Stream  │                │
│  │   Rewards       │    │ - Advantage     │                │
│  └─────────────────┘    └─────────────────┘                │
│                                   │                         │
│                          ┌─────────────────┐                │
│                          │ Replay Buffer   │                │
│                          │ - Experience    │                │
│                          │ - Memory Opt.   │                │
│                          └─────────────────┘                │
└─────────────────────────────────────────────────────────────┘

# Key Components
1.)  MemoryEfficientDataLoader
Handles large-scale HPC datasets with memory optimization:

# Supports compressed files (.gz)
# Automatic data sampling for memory constraints
# Feature engineering for HPC-specific metrics
# Synthetic data generation fallback

2.)  DigitalTwinModels
Four specialized neural networks for system prediction:

# Fault Model: Binary classification for failure prediction
# Energy Model: Regression for power consumption estimation
# Performance Model: Regression for execution time prediction
# Thermal Model: Regression for temperature forecasting

3.) DuelingDQN
Advanced deep reinforcement learning agent:

# Separates value and advantage estimation
# Dropout regularization for robustness
# Optimized for multi-objective decision making

4.)  HPCEnvironment
Simulates HPC scheduling environment:

# Job queue management
# Multi-objective reward calculation
# System state tracking
# Realistic failure scenarios

# Configuration
# Training Parameters

# RL Training
num_episodes          
batch_size           
learning_rate     
epsilon_decay     
target_update_freq    

# Multi-objective Weights
objective_weights 

# Memory Optimization
sample_fraction 
replay_buffer_size

# Model Architecture
# Digital Twin Models
input_dim 
hidden_dim 
dropout_rate 

# Dueling DQN
state_dim 
action_dim 
dqn_hidden_dim 

# Evaluation Mode
# Disable exploration for evaluation
aims.epsilon = 0.0

# Run evaluation
eval_rewards = []
for _ in range(10):
    reward = aims.train_episode(env, max_steps=100)
    eval_rewards.append(reward)

avg_reward = np.mean(eval_rewards)
std_reward = np.std(eval_rewards)
print(f"Evaluation: {avg_reward:.3f} ± {std_reward:.3f}")

# Performance Metrics
The system tracks multiple performance indicators:

1. # Training Convergence: Episode rewards over time
2. # Multi-objective Performance: Individual objective scores
3. # Digital Twin Accuracy: Prediction error rates
4. # Computational Efficiency: Training time and memory usage
5. # Scheduling Quality: Job completion rates and system utilization

# Memory Optimization
AIMS is designed for resource-constrained environments:

1. # Gradient Accumulation: Efficient batch processing
2. # Memory Cleanup: Automatic garbage collection
3. # Data Sampling: Intelligent dataset reduction
4. # Model Compression: Compact neural network architectures
5. # CUDA Optimization: GPU memory management

# Troubleshooting
# Common Issues

1.  CUDA Out of Memory
# Reduce batch size
aims.batch_size = n

# Enable memory cleanup
torch.cuda.empty_cache()

2. Dataset Loading Errors:
# Use synthetic data if datasets unavailable
data_loader = MemoryEfficientDataLoader([], sample_fraction=0.1)

3. Training Instability
# Reduce learning rate
aims.optimizer = optim.Adam(aims.q_network.parameters(), lr=5e-4)

# Increase target update frequency
aims.target_update_freq = 5

# Performance Tips
# Use GPU acceleration when available
# Adjust sample_fraction based on available memory
# Monitor training curves for convergence
# Experiment with different objective weight combinations
















