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
- LFO automations
- Additional filters used in electronic music

Ideally, Derum should be as expressive as modern digital synthesizers like Serum or Helm.

## Open ideas

- Individual note synthesizer that can change pitch (probably useful for memory, is it needed though?)
- Manually optimized non-differentiable synthesizer with identical behavior for inference (probably useful for performance, maybe, measurements necessary)

## Building

Make sure you have the latest version of `build`:

```
python3 -m pip install --upgrade build
```

Then run

```
python3 -m build
```

## Installation

I recommend installing the project in a Python 3.7 pyenv virtual environment, as DDSP seems to have issues installing for later versions as of writing.

You can install the generated wheel from the build section using pip.

## License

This project is licensed under the MIT License. Any code contribution intentionally made to the project in this repository must be licensed under the same license.
