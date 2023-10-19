import json

def load_json(path):
    with open(path) as file:
        loaded = json.load(file)
    return loaded



class LibriSpeech():
    def __init__(self):
        pass
    def get_dataset(self,subset):
        self.dataset = load_json(f"D:\Datasets\LibriSpeech\{subset}.json")
        return self.dataset