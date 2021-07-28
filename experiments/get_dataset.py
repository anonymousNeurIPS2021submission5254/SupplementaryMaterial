import numpy as np
from sklearn.preprocessing import StandardScaler as normalize
from sklearn.model_selection import train_test_split as tts

def dataset_loader(dataset_id, name, repository):
    return np.load(repository + name + str(dataset_id) + ".npy")
def prepare_dataset(dataset, train_size = 0.8, seed= False):
    kwargs = {}
    if seed or type(seed) == type(0):
        kwargs["random_state"] = seed
    X, y = dataset[:, :-1], dataset[:, -1]
    X = normalize().fit_transform(X)
    X_train, X_test, y_train, y_test = tts(X, y, train_size = train_size, **kwargs)
    return X_train, X_test, y_train, y_test
def get_dataset(dataset_id, name, repository, train_size = 0.8, seed = False):
    return prepare_dataset(dataset_loader(dataset_id, name, repository), train_size = train_size, seed = seed)