from gensim.models import KeyedVectors

from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.app import create_blueprint
from embedding_explorer.prepare.gensim import prepare_keyed_vectors

print("Loading emebddings")
keyed_vectors = KeyedVectors.load("vectors.gensim")

vocab, embeddings = prepare_keyed_vectors(keyed_vectors)

print("Creating blueprint")
blueprint = create_blueprint(vocab=vocab, embeddings=embeddings)
app = get_dash_app(blueprint)

print("Running app")
if __name__ == "__main__":
    app.run_server(debug=True)
