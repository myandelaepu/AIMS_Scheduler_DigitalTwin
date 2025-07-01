# AIMS_Scheduler_DigitalTwin

# AIMS: Uncertainty-Aware Multi-Objective HPC Scheduling with Digital Twins
# Overview
# AIMS (AI-driven Multi-objective Scheduler) is a deep reinforcement learning framework designed for adaptive HPC resource scheduling. It integrates digital twin technologies with uncertainty-aware optimization to simultaneously improve energy efficiency, system reliability, and job throughput on large-scale HPC systems.
# AIMS introduces a multi-model digital twin architecture featuring:
1.) 	LSTM-based fault prediction
2.) 	CNN-LSTM hybrid for energy forecasting
3.) 	Ensemble neural networks for performance prediction
4.) Physics-informed modeling for thermal regulation
# It also incorporates a Double Dueling Deep Q-Network (D3QN) with adaptive multi-objective weight evolution to dynamically manage trade-offs across competing HPC goals.
# Tested on real-world job traces (389,620 jobs) from Argonne’s Aurora, Polaris, Mira, and Cooley systems, AIMS outperforms state-of-the-art schedulers in:
a) +16.2% Energy Efficiency
b)	 +14.7% System Reliability
c)	 +21.6% Job Throughput
d) 	 −16.9% Performance Variability
#  Core Components
# 	Digital Twin Models
1.) PredictiveFaultModel: LSTM + Attention for fault prediction
2.) 	EnergyPredictionModel: CNN-LSTM fusion for energy forecasting
3.) 	PerformancePredictionModel: Ensemble MLPs with uncertainty estimation
4.) 	PhysicsInformedThermalModel: Domain-aware thermal dynamics estimation
# 	Scheduler Logic
1.) 	AIMSScheduler: Integrates all digital twins into a reinforcement learning loop
2.) 	DoubleDuelingDQN: Learns optimal actions using uncertainty-weighted reward aggregation
# Baseline Comparisons
1.) 	Traditional (Backfilling, HEFT, Tetris, NSGA-II)
2.) 	ML-enhanced (GreenDRL, RLSchert, Flux)
📁 Dataset
# AIMS operates on large-scale HPC logs. Included datasets (public trace-compatible format):
a) 	ANL-ALCF-MACHINESTATUS-AURORA_20250127_20250430.csv.gz
b) 	ANL-ALCF-DJC-POLARIS_20240101_20241031.csv.gz
c) 	ANL-ALCF-DJC-MIRA_20190101_20191231.csv.gz
d) 	ANL-ALCF-DJC-COOLEY_20190101_20191231.csv.gz
# Each record includes job runtime, walltime, energy use, core usage, node allocation, and exit status.
# Setup Instructions
1.) Clone the Repository
# bash
# git clone https://github.com/your-username/aims-scheduler.git
# cd aims-scheduler
2.) Install Dependencies
# bash
# pip install -r requirements.txt
# Dependencies include torch, numpy, pandas, and scikit-learn. GPU acceleration (CUDA) is optional but recommended.
3.) Add Datasets
# Place your dataset files in the root directory. Make sure the filenames match those in Config.DATASET_FILES or modify the config accordingly. You can access the dataset using this url: https://reports.alcf.anl.gov/data/index.html
# Running the Scheduler
# Run the main training loop (assuming you've wrapped AIMSScheduler.train() into a script):
# bash
# python run_aims.ipynb
# You can also evaluate baseline schedulers independently using:
# bash
# python eval_baselines.py
# (Modify run_aims.py and eval_baselines.py to suit your experimental setup.)
#  Reproducibility
# AIMS supports reproducible experiments via:
a)	Fixed random seeds (np, torch)
b) 	Dataset chunking and caching for memory efficiency
c) 	Configurable parameters via Config class
d) 	Modular architecture for easy ablation and benchmarking
#  replicate reported results:
1.)	Use the provided datasets
2.)	Maintain Config as-is
3.)	Run training for 10k+ steps (or as described in the paper)
4.)	Compare metrics: energy efficiency, reliability, throughput, variability
# Evaluation Metrics
# AIMS evaluates scheduling policies using:
# Metric	Description
# Energy Efficiency	Energy consumed per job
# System Reliability	Fault prediction accuracy
# Job Throughput	Jobs completed per time unit
# Performance Variability	Std. deviation in job completion time
# Makespan	Total time to complete all jobs
# All values are logged for comparison across schedulers.

