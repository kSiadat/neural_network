from numpy import logical_and

def error_rate(label, answer):
    is_max = answer == answer.max(axis=1)[:, None]
    single_max = is_max.sum(axis=1) == 1
    correct = logical_and(label, is_max).sum(axis=1)
    perfect = logical_and(correct, single_max)
    return (len(label) - sum(perfect)) / len(label)
