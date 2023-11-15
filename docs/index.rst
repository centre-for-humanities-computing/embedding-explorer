.. embedding-explorer documentation master file, created by
   sphinx-quickstart on Wed Nov 15 11:24:20 2023.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Getting Started
==============================================

embedding-explorer is a set of tools for interactive exploration of embedding models.
This website contains a user guide and API reference.

Installation
^^^^^^^^^^^^

You can install embedding-explorer from PyPI.

.. code-block::

   pip install embedding-explorer

Usage
^^^^^
As an example let us train a word embedding model on a corpus and then investigate the semantic relations in this model using semantic networks.
We are going to train a GloVe model on the openly available 20Newsgroups dataset.

For this we will also need glovpy, so let's install that.
Glovpy essentially has the same API as gensim's word embedding models so this example is easily extensible to gensim models.

.. code-block::
   
   pip install glovpy

Then we train an embedding model.
We do this by first loading the corpus, then tokenizing each text, then passing it to our embedding model.

.. code-block:: python

   from gensim.utils import tokenize
   from glovpy import GloVe
   from sklearn.datasets import fetch_20newsgroups
 
   # Loading the dataset
   newsgroups = fetch_20newsgroups(
       remove=("headers", "footers", "quotes"),
   ).data
   # Tokenizing the dataset
   tokenized_corpus = [
       list(tokenize(text, lower=True, deacc=True)) for text in newsgroups
   ]
 
   # Training word embeddings
   model = GloVe(vector_size=25)
   model.train(tokenized_corpus)

Now that we have trained a word embedding model,
we can start the semantic network explorer from embedding-explorer and interactively examine semantic relations in the corpus.

.. code-block:: python

   from embedding_explorer import show_network_explorer
 
   vocabulary = model.wv.index_to_key
   embeddings = model.wv.vectors
   show_network_explorer(vocabulary, embeddings=embeddings)

You will then be presented with a web application, in which you can query word association networks in the embedding model:

.. image:: _static/network_screenshot.png
    :width: 800
    :alt: Screenshot of Semantic Network.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   semantic_networks
   projection_clustering
   dashboards



Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
