
### Latest updates

Env_data for training will be updated soon.


# Synesthesia of Machines (SoM)-Enhanced ISAC Precoding for Vehicular Networks with Double Dynamics

[![IEEE](https://img.shields.io/badge/IEEE-TCOM-blue)]()
[![MATLAB](https://img.shields.io/badge/MATLAB-R2021a%2B-orange)]()
[![License](https://img.shields.io/badge/License-MIT-green)]()
[![GitHub](https://img.shields.io/badge/Open--Source-Success-success)]()

This repository contains the official implementation of the papers:

Z. Yang, S. Gao, X. Cheng and L. Yang, "Synesthesia of Machines (SoM)-Enhanced ISAC Precoding for Vehicular Networks With Double Dynamics," in IEEE Transactions on Communications, vol. 73, no. 9, pp. 7967-7984, Sept. 2025

Z. Yang, S. Gao and X. Cheng, "Doubly-Dynamic ISAC Precoding for Vehicular Networks: A Constrained Deep Reinforcement Learning (CDRL) Approach," GLOBECOM 2024 - 2024 IEEE Global Communications Conference, Cape Town, South Africa, 2024, pp. 5417-5422

**Citation:** 

```txt
@ARTICLE{10918656,
  author={Yang, Zonghui and Gao, Shijian and Cheng, Xiang and Yang, Liuqing},
  journal={IEEE Transactions on Communications}, 
  title={Synesthesia of Machines (SoM)-Enhanced ISAC Precoding for Vehicular Networks With Double Dynamics}, 
  year={2025},
  volume={73},
  number={9},
  pages={7967-7984},
  keywords={Precoding;Wideband;Training;Integrated sensing and communication;Real-time systems;Signal to noise ratio;OFDM;Accuracy;Vehicle dynamics;Time-domain analysis;Synesthesia of machine;integrated sensing and communication;deep reinforcement learning;hybrid precoding;double dynamics},
  doi={10.1109/TCOMM.2025.3549503}}
```

```txt
@INPROCEEDINGS{10901120,
  author={Yang, Zonghui and Gao, Shijian and Cheng, Xiang},
  booktitle={GLOBECOM 2024 - 2024 IEEE Global Communications Conference}, 
  title={Doubly-Dynamic ISAC Precoding for Vehicular Networks: A Constrained Deep Reinforcement Learning (CDRL) Approach}, 
  year={2024},
  volume={},
  number={},
  pages={5417-5422}}
```

## 🚀 Introduction

Integrated Sensing and Communication (ISAC) is a key technology for future vehicular networks, enabling simultaneous communication and environmental sensing. However, the dynamic nature of wireless channels and rapidly moving targets pose a doubly-dynamic challenge, making real-time precoding design extremely difficult.

This project proposes a Constrained Deep Reinforcement Learning (CDRL) framework to adaptively design ISAC precoders in doubly-dynamic scenarios, without requiring perfect prior channel or target information. The method combines:

    -Primal-Dual Deep Deterministic Policy Gradient (PD-DDPG) for constraint-aware learning

    -Wolpertinger architecture for flexible action selection with varying numbers of users

    -CNN-based feature extraction from historical channel and position states

The proposed approach achieves better sensing accuracy and communication performance compared to traditional optimization-based methods, while significantly reducing computational complexity.


## 🧠 How It Works

The system models a Constrained Markov Decision Process (CMDP) where:
    -State: Historical channel estimates and position spectra
    -Action: Updates to the analog precoding matrix
    -Reward: Negative CRLB (sensing accuracy)
    -Cost: Negative spectral efficiency (communication performance)

The agent is trained using PD-DDPG with a Wolpertinger-based action selector to handle variable user counts and complex constraints.


## ⚙️ Installation

Clone the repository and install dependencies:
```bash
git clone https://github.com/ZHYang-PKU/DRL-ISAC.git
cd DRL-ISAC
```
### Dependencies

    Python 3.8+

    PyTorch 1.12+

    NumPy, SciPy, Matplotlib

    (Optional) GPU support with CUDA


