from SPOTS.utilities.spots_10_loader import SPOT10Loader

class SPOTS:
    def __init__(self, dataset_dir="./SPOTS/dataset"):
        self.dataset_dir = dataset_dir

    def load_data(self, kind="train"):
        X, y = SPOT10Loader.get_data(dataset_dir=self.dataset_dir, kind=kind)
        X = X.astype('float32') / 255.0
        return X, y
