from sklearn.datasets import make_moons
import pickle

NUM_TRAIN_EXAMPLES = 1000
NUM_DEV_EXAMPLES = 100
NUM_TEST_EXAMPLES = 100


def create_sample_data():
    X, y = make_moons(n_samples=NUM_TRAIN_EXAMPLES, noise=0.1)
    pickle.dump({'X': X, 'y': y}, open("data/train.pkl", 'wb'))

    X, y = make_moons(n_samples=NUM_DEV_EXAMPLES, noise=0.1)
    pickle.dump({'X': X, 'y': y}, open("data/dev.pkl", 'wb'))

    X, y = make_moons(n_samples=NUM_TEST_EXAMPLES, noise=0.1)
    pickle.dump({'X': X, 'y': y}, open("data/test.pkl", 'wb'))


if __name__ == "__main__":
    create_sample_data()
