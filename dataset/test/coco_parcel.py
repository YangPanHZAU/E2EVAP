from .base import Dataset
from skimage import io
class CocoTestDataset_parcel(Dataset):
    def read_original_data(self, path):
        img = io.imread(path)
        return img