# About
Unsupervised speech disentanglement using GANs

Inspired by [this paper](https://arxiv.org/abs/2005.12963).

The model employs a content extractor, speaker extractor, GAN generator and discriminator. 

The model learns to disentangle a spectrogram into it's content and speaker then reconstructing the spectrogram. So it is capable of extracting speaker embeddings, speech content and performing voice conversion. 

Reconstruction example from an early stage of the training.
ground truth (left) vs reconstructed (right).
<img src="/main/early_example.png" alt="comp" >
