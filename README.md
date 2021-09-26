# derum

Differentiable electronic music synthesizer for use with TensorFlow.

This project is currently in development but is usable. More features are to be expected.

## Why?

The idea behind derum is to make neural networks generate a "high-level" representation of (electronic) music instead of direct audio data. This is achieved by creating what effectively is a regular electronic music synthesizer, except that it only uses differentiable transformations. This approach should be helpful to solve the dilemma between abundant sheet music data in restricted genres versus scarce sheet music data in broad genres.

This is built upon [DDSP](https://github.com/magenta/ddsp/) from the Magenta group at Google. I hope Derum will reduce the output space to offer benefits similar to what DDSP allowed for general-purpose synthesis.

## State of the project

I've been exploring the methods during the summer of 2021. A first version of the synthesizer used a very naive solution for audio envelope using only a convolution with a free filter. This proved to be too unconstrained for many use cases, so I have implemented a new envelope generation strategy that uses a more traditional ADSR. The difficult part was to keep it differentiable and parallel when most ADSR implementation techniques use control-flow-heavy recurrent implementations.

The implementation of the new method is complete. I am currently measuring its benefits by applying it to the problem of reconstructing tracks of polyphonic music.

## Roadmap

- Better phase control (phase currently is based on global time)
- Additional filters used in electronic music

Ideally, Derum should be as expressive as modern digital synthesizers like Serum or Helm.

## Open ideas

- Individual note synthesizer that can change pitch (probably useful for memory, is it needed though?)
- Manually optimized non-differentiable synthesizer with identical behavior for inference (probably useful for performance, maybe, measurements necessary)

## Installation

In order to install DDSP, I recommend setting up a pyenv with Python 3.7.
To install Derum, simply install latest `pip`, `tensorflow` and `ddsp` (in that order) via pip. Derum is not currently shipped in a wheel, so for now you should import the project's source directly.

## License

This project is licensed under the MIT License. Any code contribution intentionally made to the project in this repository must be licensed under the same license.
