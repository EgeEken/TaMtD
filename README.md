# TaMtD
Give a VLM Model a decoder, and you'll have to sample frames for a lifetime, Teach a Model to Decode (TaMtD) and you never have to sample again (hopefully)

The idea for this project is to develop a new technique for video / image comprehension for VLM's where instead of going through this process of:

```Encoded media file -> Media Decoder -> Vision Encoder -> LLM -> Output```

We can instead just teach the VLM to decode the media file itself, and do this:

```Encoded media file ->       TaMtD ViT Decoder         -> LLM -> Output```

The core idea is that compressed media files for videos and images already have a lot of structure to them hidden somewhere in their bitstream, and they are made specifically to be decoded by a decoder algorithm and maintain all semantic visual information while cutting down the bit count. Meanwhile ViT's serve the purpose of extracting semantic information from uncompressed image matrices, and that is actually a much more difficult task than simply decoding compressed media files, so if we can teach a ViT to simultaneously decode and extract semantic information from compressed media files, we can skip the decoding step entirely which saves some overhead, and if this bitstream reading can be made efficient enough through a very high rate of tokenization, we can achieve video comprehension with full frame coverage, which is something that is completely impossible with the traditional approach.

Steps i have in mind for this project are:

1- Preliminary testing to see if i can teach a model to decode a media file, this will be a small pipeline where the model will simply be required to take in the bitstream of a compressed media file, and output the uncompressed image matrix

- The data collection for this will be trivially easy since literally every single compressed media file has the "ground truth" for this already, which is just the media itself, so no manual labeling, no unsupervised learning, no reinforcement learning, no complicated loss functions or distillation processes, just a simple supervised learning problem.

- The model design will be a larger challenge, but still probably not too much of a challenge, overall it has to simply take in a 1D sequence of bits and output a 3D matrix of pixel values (or 4D for videos), can take inspiration from image generation models for this.

- It might be a good idea to design the architecture in a semi-auto-encoder way so that there's an intermediary step where the model first generates an even further compressed representation of the media file, before attempting to recreate it, this way when the final TaMtD ViT is created, we can simply cut out the generator since it won't be needed, and just have the "decoder" part of the model

2- A tokenizer will be needed to convert the bitstream of the compressed media files into tokens that can be fed into the model, this tokenizer will have to be custom made and trained since obviously there is no existing tokenizer for tokenizing media file bitstreams. But i feel like simply running a standard tokenizer creation algorithm on the bitstreams of a relatively large dataset of media files will be enough since the logic of finding the most common sequences of bits and making those into tokens should be the same regardless of the fact that these bitstreams are for media files and not text.

3- Once the preliminary testing is done, we can proceed to train the full TaMtD ViT model on a larger dataset of compressed media files, ensuring that it can effectively decode and extract semantic information from the bitstreams. Using regular VLM evaluation methods like zero shot classification, context retrieval, visual question answering, captioning etc, we can compare the quality of this model to regular vlms.

Expectations:

- If this works, and the final model can effectively decode and read from the full bitstreams, this could be great news for video comprehension since it would cut out the need for frame sampling, and allow for much more in depth video comprehension since the model would have access to the full video instead of just a few sampled frames
