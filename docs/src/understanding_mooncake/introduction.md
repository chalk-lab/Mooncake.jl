# Introduction

The point of Mooncake.jl is to perform reverse-mode algorithmic differentiation (AD).
The purpose of this section is to explain _what_ precisely is meant by this, and _how_ it can be interpreted mathematically.
1. we recap what AD is, and introduce the mathematics necessary to understand it,
1. explain how this mathematics relates to functions and data structures in Julia, and
1. how this is handled in Mooncake.jl.

Since Mooncake.jl supports in-place operations / mutation, these will push beyond what is encountered in Zygote / Diffractor / ChainRules.
Consequently, while there is a great deal of overlap with these existing systems, you will need to read through this section of the docs in order to properly understand Mooncake.jl.

# Who Are These Docs For?

These are primarily designed for anyone who is interested in contributing to Mooncake.jl.
They are also hopefully of interest to anyone who is interested in understanding AD more broadly.
If you aren't interested in understanding how Mooncake.jl and AD work, you don't need to have read them in order to make use of this package.

# Prerequisites and Resources

This introduction assumes familiarity with the differentiation of vector-valued functions -- familiarity with the gradient and Jacobian matrices is a given.

In order to provide a convenient exposition of AD, we need to abstract a little further than this and make use of a slightly more general notion of the derivative, gradient, and "transposed Jacobian".
Please note that, fortunately, we only ever have to handle finite dimensional objects when doing AD, so there is no need for any knowledge of functional analysis to understand what is going on here.
The required concepts will be introduced here, but I cannot promise that these docs give the best exposition -- they're most appropriate as a refresher and to establish notation.
Rather, I would recommend a couple of lectures from the "Matrix Calculus for Machine Learning and Beyond" course, which you can find [on MIT's OCW website](https://ocw.mit.edu/courses/18-s096-matrix-calculus-for-machine-learning-and-beyond-january-iap-2023/), delivered by Edelman and Johnson (who will be familiar faces to anyone who has spent much time in the Julia world!).
It is designed for undergraduates, and is accessible to anyone with some undergraduate-level linear algebra and calculus.
While I recommend the whole course, Lecture 1 part 2 and Lecture 4 part 1 are especially relevant to the problems we shall discuss -- you can skip to 11:30 in Lecture 4 part 1 if you're in a hurry.
