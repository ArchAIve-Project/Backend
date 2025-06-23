from addons import ModelContext, ModelStore, ASTracer, ASReport
from cnnclassifier import CNNModel, ImageClassifier

ModelStore.setup(
    cnn=ImageClassifier.load_model
)