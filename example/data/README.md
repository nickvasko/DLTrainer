## Data Folder Introduction

The data folder stores all of your training datasets: train, dev, and test. You can place as many files in here as you like. You will setup how DLTrainer
interacts with your data folder through your custom dataset class (see dataset.py) for an example. This allows you to customize how your data is handled.

## Recommendations

Preprocess your data only once. For example, in dataset.py the code looks for cache_file. If it does not exist it creates a new dataset, otherwise it just
loads the previous dataset.

```python
class SimpleDataset(Dataset):
    def __init__(self, args, file_type):
        cache_file = f'{args.data_dir}/{file_type}.pkl'
        if not os.path.exists(cache_file):
            print(f'dataset not found in {cache_file}. creating it now.')
            create_sample_data(100000)

        data = pickle.load(open(cache_file, 'rb'))

        self.inputs = data['X']
        self.labels = data['y']
        assert len(self.inputs) == len(self.labels), "inputs and labels need to be the same length."

```
