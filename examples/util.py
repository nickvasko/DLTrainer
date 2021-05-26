from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
import pickle

NUM_TRAIN_EXAMPLES = 100000


def create_sample_data():
    X, y = make_moons(n_samples=NUM_TRAIN_EXAMPLES, noise=0.1)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=1/9, random_state=42)
    pickle.dump({'X': X_train, 'y': y_train}, open("data/train.pkl", 'wb'))
    pickle.dump({'X': X_val, 'y': y_val}, open("data/dev.pkl", 'wb'))
    pickle.dump({'X': X_test, 'y': y_test}, open("data/test.pkl", 'wb'))


if __name__ == "__main__":
    create_sample_data()
