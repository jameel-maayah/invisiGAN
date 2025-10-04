# invisiGAN

## Abstract
We propose a methodology to embed binary sequences into the output of generative AI models, in particular Generative Adversarial Networks (GANs)[1], in ways imperceptible to the eye and resilient to image compression. Our technique conditions the generator network on a bit vector in addition to random noise, and the generator’s primary loss gradient will be supplied by the discriminator network. In addition, we incorporate a third “probe” network that attempts to decode the binary vector encoded by images (both original and compressed) created by the generator. This gives us a secondary training target, namely the accurate recovery of the original binary sequence. The job of the generator is to balance the “readability” of generated images whilst preserving its approximation of the original dataset.

We would like to opt in for the Outstanding Project award.

## 1. Literature

Generative AI models are capable of reproducing artificial images that replicate existing databases[1]. Incremental developments of these models have allowed for the conditioning of the output of traditional generative networks by representing class labels or other features as additional input to the generator model[2]. We now look at a specific application of the GAN with its usage in hiding arbitrary binary data in images while keeping the changes hard to detect[3]. We will use various methods outlined by Claude Shannon in his “Mathematical Theory of Communication”[4] including rate-distortion theory and mutual information. 

### Dataset Description
The MNIST dataset is a small dataset which contains over 70,000 greyscale images of hand-drawn digits 0 through 9. 
### Dataset Link 
https://www.kaggle.com/datasets/hojjatk/mnist-dataset

## 2. Problem and Motivation

### Problem 
In recent years, the addition of watermarking into generative models has been explored in models like SteganoGAN[3], which embed messages into the input image which are undetectable to users but can be decrypted with a trained model, usually a CNN. However, as images are often compressed when they are shared online, these watermarks are lost when the exact pixel values of the image changes. 
### Motivation
As AI generated content becomes increasingly indistinguishable from reality, the need for users to be confident in the authenticity of the images they see and share online is increasingly important. However, the popularity of these tools derives, fundamentally, from their ability to approximate reality. As such, any method of embedding a “watermark” into the output of a generative model must not disrupt the user experience. Since images shared online mostly undergo compression, it is critical that we preserve watermarkability through compression.

## 3. Methodology

We adopt a traditional cGAN training framework that conditions the generator input on a random binary sequence instead of class labels. The generator network is trained with two objectives: an adversarial objective supplied by the discriminator, which attempts to distinguish images that were generated versus sampled from the dataset; and an auxiliary objective supplied by the decoder, which attempts to decode the binary message encoded in the generated image. The discriminator loss is calculated via binary cross-entropy between the two classes of data (fake and real), whereas the decoder loss is calculated via binary cross-entropy between each logit of the predicted and original binary sequences. To provide a gradient to the generator for optimization, we simply weigh these losses equally. 

### Data Preprocessing Techniques
We will partition the dataset into training, validation, and test sets to prevent overfitting of the generator and discriminator. Before inputting grayscale images to the discriminator and decoder, we normalize them to pixel values between 0 and 1 for stabler gradients. In order to encourage compression resistance, we will pass generated images through JPEG transformations during training, requiring the decoder to recover the encrypted bits even after compression artifacts are added.

### ML Models/Algorithms
We will use convolutional neural networks for the discriminator and decoder, and stacked transposed convolutional layers for the generator. All of these will be created and optimized via gradient descent using built-in pytorch functions and classes (torch.nn.Linear, torch.nn.Conv2d, torch.optim.Adam, etc.), with error signals also computed using torch.nn.BCELoss.

### Supervised Learning Method
Our methodology uses supervised learning methods because both the discriminator and the decoder are trained with labeled data (real vs. fake images, and binary sequences, respectively). The generator is simultaneously trained using signals from these explicit, supervised objectives.

## 4. Results/Discussion
We evaluate our system using 3 metrics, Peak Signal-to-Noise Ratio (PSNR), which measures the visual similarity between the cover and steganographic images; Bit Recovery Accuracy to measure robustness under compression; and Exact Match Rate to quantify the percentage of payloads decoded perfectly without error. Our goal is to maintain a high PSNR to ensure high perceptual quality, a high bit rate accurate (>90%), and maximize Exact Match Rate across the dataset. In addition, we perform an ablation study by varying the length of embedded message length, using 0 as a control group, to analyze the trade off between payload size and robustness of image generation. We expect, essentially, to generate visually unaltered images with a highly accurate recovery of the embedded steganographic message.

### Contributions
Jameel Maayah: Abstract, Methods
Caleb Rieck: Problem/Motivation, Results/Discussion
Srikar Satluri: Literature Review, References
Ajinkya Argonda: Slides, Video
Anish Vallabhaneni: Slides

## 5. References

[1] I. J. Goodfellow et al., “Generative Adversarial Networks,” arXiv.org, Jun. 10, 2014. https://arxiv.org/abs/1406.2661
‌[2] M. Mirza and S. Osindero, “Conditional Generative Adversarial Nets,” arXiv.org, 2014. https://arxiv.org/abs/1411.1784
‌[3] K. A. Zhang, A. Cuesta-Infante, L. Xu, and K. Veeramachaneni, “SteganoGAN: High Capacity Image Steganography with GANs,” arXiv:1901.03892 [cs, stat], Jan. 2019, Available: https://arxiv.org/abs/1901.03892
[4] C. E. Shannon and W. Weaver, “A Mathematical Theory of Communication,” Bell System Technical Journal, vol. 27, no. 4, pp. 623–656, Oct. 1949, doi: https://doi.org/10.1002/j.1538-7305.1948.tb00917.x.
‌


‌







