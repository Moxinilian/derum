# derum

Differentiable electronic music synthesizer for use with TensorFlow.

This project is currently in development and is not complete yet.

## Why?

The idea behind derum is to make neural networks generate a "high-level" representation of (electronic) music instead of direct audio data. This is achieved by creating what effectively is a regular electronic music synthesizer, except that it only uses differentiable transformations. This approach should be helpful to solve the dilemma between abundant sheet music data in restricted genres versus scarce sheet music data in broad genres.

This is built upon [DDSP](https://github.com/magenta/ddsp/) from the Magenta group at Google. I hope Derum will reduce the output space to offer benefits similar to what DDSP allowed for general-purpose synthesis.

## State of the project

I've been explorating the methods during the summer of 2021. A first version of the synthesizer used a very naive solution for audio envelope using only a convolution with a free filter. This proved to be too unconstrained for many use cases, so I am currently implementing a new envelope generation strategy that uses a more traditional ADSR. The difficult part is to keep it differentiable.

The implementation of the new method is complete. I am currently measuring its benefits by applying it to the problem of reconstructing tracks of polyphonic music.

## Roadmap

- ADSR-based envelope generation.
- Better phase control (phase currently is based on global time)
- Additional filters used in electronic music

Ideally, Derum should be as expressive as modern digital synthesizers like Serum or Helm.

## License

This project is licensed under the MIT License. Any code contribution intentionally made to the project in this repository must be licensed under the same license.
