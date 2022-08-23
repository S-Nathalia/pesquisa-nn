from math import factorial

def complex(net):
    complexity = 1
    for i in range(1, len(net)-1):
        complexity *= (net[i] // net[0]) ** net[0]
    aux = 0
    for j in range(net[0]):
        aux += factorial(len(net) - 1) / factorial(len(net) - j)
    complexity *= aux

    return complexity

def normalization(train, test):
    features = (train.shape[1])

    for i in range(features):
        min_t = train.iloc[:, i].min()
        max_t = train.iloc[:, i].max()

        train.iloc[:, i] = (train.iloc[:, i]-min_t)/(max_t-min_t)
        test.iloc[:, i] = (test.iloc[:, i]-min_t)/(max_t-min_t)

    return train, test


