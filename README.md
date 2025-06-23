# Gradients Protection in Federated Learning for Biometric Authentication

---

## Overview

This repository contains the code and experimental setup for our research paper on securing federated learning systems used for biometric authentication against gradient inversion attacks.

We focus on protecting sensitive biometric data — including face and fingerprint biometrics — in decentralized, privacy-preserving machine learning environments. Our modular framework integrates advanced data augmentation and multi-modal biometric fusion to enhance authentication robustness while mitigating privacy risks.

---

## Research Topic

**Gradients Protection in Federated Learning for Biometric Authentication**

---

## Key Contributions

- Developed a modular, decentralized machine learning framework designed to secure biometric authentication workflows against gradient inversion attacks in federated learning settings.

- Implemented advanced data augmentation techniques combined with multi-biometric data integration (facial recognition and fingerprint data) to improve privacy preservation and authentication accuracy.

- Conducted extensive experimental and theoretical evaluations demonstrating the framework's superior performance compared to state-of-the-art defenses, significantly enhancing security, scalability, and operational viability.

---

## Repository Structure

/data/ # Biometric datasets and pre-processing scripts
/models/ # Model architectures (ResNet, ViT, etc.)
/augmentations/ # Data augmentation pipelines
/attacks/ # Gradient inversion and leakage attack implementations
/experiments/ # Experimental scripts and evaluation pipelines
/utils/ # Utility functions and helpers
/configs/ # Configuration files for training and experiments
/results/ # Logs, metrics, and generated outputs



Usage
Prepare biometric datasets (face and fingerprint) and place them in the /data/ folder following the specified structure.
Configure experiment parameters in /configs/.
Train models and run experiments:
python experiments/train.py --config configs/your_config.yaml
Evaluate defenses against gradient inversion attacks:
python attacks/gradient_inversion_attack.py --model_path models/your_trained_model.pth
Results and Evaluation

Please refer to the /results/ folder for detailed metrics, visualizations, and comparative analysis demonstrating the effectiveness of our proposed gradient protection methods.

Citation

If you use this work or code, please cite our paper
