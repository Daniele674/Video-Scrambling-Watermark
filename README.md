# Scrambling and Watermarking in Videos

This repository contains the source code for the project "Scrambling and Watermarking in Videos", developed as part of the Data Compression exam for the Master's Degree in Information Security at the University of Salerno (Academic Year 2024-2025).

The project aims to develop an advanced system for protecting digital content based on two complementary techniques:

- **Reversible Scrambling**: Selective obfuscation of sensitive regions (such as faces) in a video by manipulating the coefficients of the Discrete Cosine Transform (DCT). The techniques employed—either through permutation or sign inversion—allow controlled obfuscation and the subsequent restoration of the original content via a descrambling process.
- **Invisible Watermarking**: Embedding an encrypted digital watermark (using methods based on DWT, DCT, and SVD along with AES encryption) to ensure the authenticity and integrity of the content. This watermark contains encrypted metadata necessary for recovering the original data, accessible only to authorized users.

## Main Features

- **Selective and Reversible Obfuscation**: Protects faces and other sensitive areas by applying scrambling techniques in the frequency domain.
- **Robust Digital Watermarking**: Incorporates an invisible watermark capable of withstanding manipulations and compression, embedding essential encrypted metadata for the descrambling process.
- **Parallel Processing**: Utilizes multiprocessing to accelerate the processing of video frames.
- **Integration of Specialized Libraries**: Built in Python using libraries such as OpenCV, Mediapipe, NumPy, Cryptography, FFmpeg, jpeglib, and ReedSolomon, ensuring a modular and efficient solution.

## Technologies and Libraries Used

- **Python 3.x**
- **OpenCV**: For video frame processing and manipulation.
- **Mediapipe**: For detecting facial landmarks to identify regions of interest.
- **jpeglib**: For manipulating DCT coefficients in JPEG images.
- **NumPy**: For numerical computations and array management.
- **Cryptography**: For AES-based encryption and decryption.
- **FFmpeg**: For creating and modifying video files.
- **ReedSolomon (reedsolo)**: For error correction in the watermarking process.
- **Multiprocessing**: For parallel processing to improve performance.

## System Architecture

The system workflow is divided into two main phases:

1. **Scrambling (Obfuscation)**
   - **Video Acquisition**: The input video is read and split into individual frames.
   - **Sensitive Region Detection**: Mediapipe is used to detect faces and other areas of interest.
   - **Scrambling Application**: Obfuscation is applied using one of the following methods:
     - **Permutation**: Randomly reorders the DCT coefficients.
     - **SignFlip**: Inverts the sign of a subset of the DCT coefficients.
   - **Watermark Embedding**: A digital watermark containing encrypted metadata is embedded into the video, which is essential for the descrambling process.

2. **Descrambling (Recovery)**
   - **Watermark Extraction**: The embedded watermark is decoded to retrieve the necessary parameters.
   - **Original Content Restoration**: The inverse transformation is applied to the DCT coefficients to reconstruct the original video.

## Requirements

- **Python 3.x**
- **FFmpeg** must be installed and configured in the system PATH.
- All required libraries are listed in the `requirements.txt` file, which includes OpenCV, Mediapipe, NumPy, Cryptography, reedsolo, and other dependencies.

## Installation

1. **Clone the Repository:**
   ```bash
   git clone https://github.com/your-username/scrambling-watermarking.git
   cd scrambling-watermarking
