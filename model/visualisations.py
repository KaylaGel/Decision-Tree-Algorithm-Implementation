import matplotlib.pyplot as plt

def plot(name, test_accuracy, training_accuracy): 

    plt.title(f"{name} - Testing VS Training data accuracy")
    plt.xlabel("Time (iterations)")
    plt.ylabel("Raw Accuracy")
    plt.plot(test_accuracy, label = 'test data accuracy');
    plt.plot(training_accuracy, label = 'training data accuracy');
    plt.legend()
    plt.show()
    