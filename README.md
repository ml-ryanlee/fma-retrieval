# Remix-Proof Retrieval: Robust Audio Encoders for Detecting Copyright Infringement

This repository contains the implementation and research for our audio retrieval system designed to detect copyright infringement in digital music, even when audio has been modified through various transformations.

## Overview

The rapid proliferation of digital music platforms necessitates robust methods for detecting copyright infringement, yet traditional approaches relying on Fast Fourier Transforms (FFTs) struggle with nuanced perturbations and diverse audio representations. In this work, we evaluate music information retrieval (MIR) performance under adversarial augmentations such as pitch shifting, background noise, and equalization.

We fine-tune audio encoders using contrastive learning techniques inspired by SimCLR and assess two architectures—CLAP and Music2Latent—on a benchmark retrieval task. By applying a suite of targeted audio augmentations during training, we improve the models' ability to generate stable embeddings, achieving higher retrieval accuracy in noisy or distorted conditions. Our experiments demonstrate that the Music2Latent framework significantly enhances robustness and retrieval quality over zero-shot baselines.

## Motivation

Copyright infringement detection is critical for protecting intellectual property in the digital music industry. Existing methods struggle with modern challenges like audio perturbations and diverse representations, leading to missed infringements or false positives. Our approach offers a more robust and fine-grained solution, ensuring creators are fairly compensated and copyright laws are upheld in an evolving digital landscape.

Additionally, our techniques can benefit high-quality jazz music generation. It reduces production costs for fresh background music—ideal for businesses like restaurants—and enables composers to rapidly prototype and showcase their creative ideas to potential clients.

## Dataset

To support our approach for copyright infringement detection, we developed a structured dataset designed specifically for retrieval-based matching. The dataset was sourced from the `benjamin-paine/free-music-archive-small` collection on Hugging Face, containing a total of 7,916 audio tracks. Each source song, which we treat as a copyrighted song, has an approximate duration of 30 seconds and is evenly distributed across 8 genres, ensuring a diverse representation of musical styles.

To simulate realistic copyright infringement scenarios, we extracted a random span of 5-second clips from each track to serve as queries. This approach reflects real-world situations where only short segments of an audio piece may be available for comparison. To benchmark song identification against evasion strategies, we further augmented these five-second clips to serve as disguised copies of the copyrighted songs.

Our benchmark is available on [HuggingFace](https://huggingface.co/datasets/ryanleeme17/free-music-archive-retrieval).

## Features

- Robust audio encoding for copyright infringement detection
- Fine-tuned models resistant to common audio perturbations
- Evaluation framework for music information retrieval
- Benchmarking against adversarial augmentations
- Implementation of contrastive learning techniques

## Contributors (Alphabetical)

- [Abhir Karande](mailto:akarande@usc.edu)
- [Ayush Goyal](mailto:ayushgoy@usc.edu)
- [Harrison Pearl](mailto:hpearl@usc.edu)
- [Matthew Hong](mailto:hongmm@usc.edu)
- [Ryan Lee](mailto:ryantlee@usc.edu)
- [Spencer Cobb](mailto:srcobb@usc.edu)
- [Yi-Chieh Chiu](mailto:ychiu443@usc.edu)

## Citation

If you use this code or dataset in your research, please cite our work:

@misc{lee2025remixproof,
  title={Remix-Proof Retrieval: Robust Audio Encoders for Detecting Copyright Infringement},
  author={Lee, Ryan* and Chiu, Yi-Chieh* and Karande, Abhir* and Goyal, Ayush and Pearl, Harrison and Hong, Matthew and Cobb, Spencer},
  year={2025},
  note={*Equal contribution},
  publisher={GitHub},
  howpublished={\url{https://github.com/username/remix-proof-retrieval}}
}

