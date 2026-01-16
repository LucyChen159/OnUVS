# OnUVS: Online Motion Transfer for Ultrasound Video Synthesis

This repository provides the official implementation of **OnUVS**, an online motion transfer framework for high-fidelity ultrasound video synthesis, as described in our journal submission.

## Overview

OnUVS is designed to synthesize realistic ultrasound videos by transferring motion patterns from a driving video to a static source image.  
The framework explicitly decouples **content**, **texture**, and **motion**, and incorporates both local keypoint-based deformation and global motion modeling, together with an online learning strategy to enhance temporal coherence.

The method is particularly tailored to ultrasound imaging scenarios, where strong local deformation, speckle noise, and limited global texture consistency pose challenges for generic video generation models.

## Key Features

- Keypoint-driven local motion learning with supervised and unsupervised keypoints  
- Global motion modeling for coherent large-scale deformation  
- Online learning strategy for improving long-term temporal consistency  
- Support for both cardiac and pelvic ultrasound video synthesis  
- Evaluation on normal and pathological cases, with downstream clinical tasks


