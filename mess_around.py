import dash_mantine_components as dmc
from dash_extensions.enrich import DashBlueprint
from gensim.models import KeyedVectors

from embedding_explorer.app import get_dash_app
from embedding_explorer.blueprints.dashboard import create_dashboard
from embedding_explorer.components.model_card import create_card
from embedding_explorer.prepare.gensim import prepare_keyed_vectors

MODEL_NAMES = [
    "glove-twitter-50",
    "glove-twitter-25",
    "glove-wiki-gigaword-50",
    # "word2vec-google-news-300",
]

# Download vectors if need be
# from gensim import downloader
# for model_name in MODEL_NAMES:
#     keyed_vectors = downloader.load(model_name)
#     keyed_vectors.save(f"dat/{model_name}.gensim")

# Loading embeddings
models = {}
for model_name in MODEL_NAMES:
    keyed_vectors = KeyedVectors.load(f"dat/{model_name}.gensim")
    model = prepare_keyed_vectors(keyed_vectors)
    models[model_name] = model

blueprint, register_pages = create_dashboard(models)

# print("Creating blueprint")
# blueprint = create_blueprint(vocab=vocab, embeddings=embeddings)

app = get_dash_app(blueprint)
register_pages(app)

print("Running app")
if __name__ == "__main__":
    app.run_server(debug=True)
