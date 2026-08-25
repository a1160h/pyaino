pyaino is a deep learning framework designed for practical AI development while keeping the transparency and directness of NumPy at its core.

A fundamental design principle of pyaino is that **the model should not be locked into the framework**. Although pyaino provides automatic differentiation and GPU acceleration for training, the forward computation of a model remains expressed in ordinary NumPy/CuPy-style array operations. As a result, a trained model can be reconstructed from its model definition and checkpoint — together with a tokenizer or other preprocessing components when needed — and executed with NumPy alone, without requiring the pyaino runtime.

This portability does not depend on exporting the model to an intermediate graph format or converting it to a framework-specific inference representation. The computational structure of the original model remains visible as ordinary Python and array operations. In this sense, pyaino is intended not only to make models easy to build and train, but also to make them **portable, inspectable, and reproducible outside the framework itself**.

pyaino provides automatic differentiation in a define-by-run manner while adopting the same matrix multiplication convention as mathematics, avoiding the unnecessary confusion that can arise for users familiar with numerical computation or mathematical notation.

Another distinctive feature is its treatment of higher-order derivatives. pyaino computes them according to their mathematical structure; for example, the second derivative of a linear function correctly evaluates to zero rather than being represented by an artificial residual computation graph.

For practical model development, pyaino also provides ready-to-use modules for image processing, language modeling, transformers, normalization, convolutional networks, diffusion models, and other common deep-learning tasks, allowing substantial models to be constructed with relatively little code.

At the same time, pyaino keeps the underlying computation explicit. Each primitive operation defines its forward and backward propagation directly, so unnecessary computation graphs are not retained. This helps reduce resource usage while making the behavior of models easier to inspect and verify.

The goal of pyaino is therefore not to hide numerical computation behind a framework, but to provide the machinery needed for modern deep learning **without losing direct access to the computation itself**.
