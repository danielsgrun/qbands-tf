# qbands-tf

Calculation of energy band formation on a four-well quantum integrable model, whose hamiltonian is given by:

<a href="https://www.codecogs.com/eqnedit.php?latex=H&space;=&space;-U(N_1&space;&plus;&space;N_3&space;-&space;N_2&space;-&space;N_4)^2&space;-&space;\frac{J}{2}\left[(a_1&space;&plus;&space;a_3)(a_2^\dagger&space;&plus;&space;a_4^\dagger)&space;&plus;&space;\text{h.c.}&space;\right]" target="_blank"><img src="https://latex.codecogs.com/gif.latex?H&space;=&space;-U(N_1&space;&plus;&space;N_3&space;-&space;N_2&space;-&space;N_4)^2&space;-&space;\frac{J}{2}\left[(a_1&space;&plus;&space;a_3)(a_2^\dagger&space;&plus;&space;a_4^\dagger)&space;&plus;&space;\text{h.c.}&space;\right]" title="H = -U(N_1 + N_3 - N_2 - N_4)^2 - \frac{J}{2}\left[(a_1 + a_3)(a_2^\dagger + a_4^\dagger) + \text{h.c.} \right]" /></a>

In the previous equation, U and J characterize, respectively, the "strength" of interaction and hopping; "h.c." means "hermitian conjugate". The code also contains a brief application of deep-learning, which is capable of reproducing the non-highly-degenerate parts of the spectrum with a fairly simple neural network.

By running the code, the output will be both the quantum-mechanically-evaluated energies with respect to U/J (fixed J, varying U) and the prediction (and training/testing statistics) of the neural network.

Please, refer to arXiv:2004.11987 for more details on the model, such as its importance in the context of interferometry and metrology.
