from gensim.models import KeyedVectors

from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.app import create_blueprint
from embedding_explorer.prepare.gensim import prepare_keyed_vectors

# Download vectors if need be
# from gensim import downloader
# keyed_vectors = downloader.load("glove-twitter-50")
# keyed_vectors.save("dat/vectors.gensim")

print("Loading emebddings")
keyed_vectors = KeyedVectors.load("dat/vectors.gensim")

vocab, embeddings = prepare_keyed_vectors(keyed_vectors)

print("Creating blueprint")
blueprint = create_blueprint(vocab=vocab, embeddings=embeddings)
app = get_dash_app(blueprint)

print("Running app")
if __name__ == "__main__":
    app.run_server(debug=True)
