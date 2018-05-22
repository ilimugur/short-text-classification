# short-text-classification

This repo contains a command line tool that helps train different short text classification models using different options. Created mainly as part of a thesis work on the subject, this tool helps users train deep learning models in a couple of papers referenced below, and enables users to be able to easily experiment with training those models using different loss functions, optimizers, datasets or word embeddings.

The models implemented are the ones described in the papers referenced below. I may implement a few more papers as I keep on expanding my thesis work.

- [J. Y. Lee, and F. Dernoncourt, "Sequential short text  classification with  recurrent  and  convolutional  neural  networks,"  arXiv  preprint arXiv:1603.03827, 2016.](https://arxiv.org/abs/1603.03827)

- [H. Kumar, A. Agarwal, R. Dasgupta, S. Joshi, A. Kumar, "Dialogue Act Sequence Labeling using Hierarchical encoder with CRF,"  arXiv  preprint arXiv:1709.04250, 2017.](https://arxiv.org/abs/1709.04250)

Note that, for a while, these models may yield a much lower training, validation and testing accuracy due to implementation issues.


### Requirements
This tool requires quite a few libraries as prerequisites. It uses [Keras](https://keras.io/) and, naturally, all its prerequisites. It also requires [TensorFlow](https://www.tensorflow.org/).

Additionally, although the tool supports taking a couple of different word embeddings as input, the word embeddings themselves should be separately downloaded, and if necessary unzipped, as well.

Similarly, the datasets supported should be downloaded and/or unzipped by the user separately. Currently, the only supported dataset is [SwDA](https://web.stanford.edu/~jurafsky/ws97/), and the dataset is also included in the swda submodule inside the repo. However, the user should unzip the file into a desired directory, as the tool itself does not handle the unzipping operation.

Finally, if `--save-model` option is to be used, the Python module [h5py](https://pypi.python.org/pypi/h5py) is required, and it can be installed via [pip](https://pip.pypa.io/en/stable/installing/).

##### Word Embeddings:
- [Word2Vec](https://code.google.com/archive/p/word2vec/): Download file from [here](https://drive.google.com/file/d/0B7XkCwpI5KDYNlNUTTlSS21pQmM/edit?usp=sharing)
- [GloVe](https://nlp.stanford.edu/projects/glove/): Download file from [here](http://nlp.stanford.edu/data/glove.twitter.27B.zip)

### Sample usage
Tool may be used in a couple of ways.

Using `--loss-functions` or `--optimizers` options, it may be used to list the loss functions and optimizers supported by the Keras version you are using.
```console
foo@bar:~$ python core.py --loss-functions
```

Using `--embeddings` or `--datasets` options, it may be used to list the list of supported word embeddings and datasets, respectively.
```console
foo@bar:~$ python core.py --embeddings
```

Similarly, using `--models` option, the list of implemented models may be printed.
```console
foo@bar:~$ python core.py --models
```

Finally, to train a specific model by specifying a dataset, an embedding, a loss function and an optimizer, you may use a command similar to the one given below.

```console
foo@bar:~$ python core.py --model Lee-Dernoncourt
                          --dataset SwDA <path_to_SwDA_dataset_directory>
                          --embedding GloVe <path_to_GloVe_embedding_file>
```

For a more detailed description of the capabilities of the tool, use `--help` option.

```console
foo@bar:~$ python core.py --help
```
