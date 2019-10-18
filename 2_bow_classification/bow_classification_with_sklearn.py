import csv
import random
import re
import sys
import os
import numpy as np
from tqdm import tqdm
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression, Perceptron
from sklearn.naive_bayes import MultinomialNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn import svm
from sklearn.feature_extraction.text import TfidfTransformer

"""

# Bag-of-Words Classification with scikit-learn

In this task, you will implement the Bag-of-Words model and text classification in Python.

Implement two methods:
- `preprocess_and_split_to_tokens(sentences) -> tokens_per_sentence`
- `create_bow(sentences, vocab, msg_prefix) -> (vocab, bow_array)`

## Instruction
* See skeleton codes below for more details.
* Do not remove assert lines and do not modify methods that start with an underscore.
* Do not use bag-of-words function implemented in sklearn.
* Before submit your code in KLMS, please change the name of the file to your student id (e.g., 2019xxxx.py).
* Prediction accuracy for unknown test samples (i.e., we do not give them to you) will be your grade.
* Fine-tune your model to get higher accuracy, you can freely choose the pre-processing, model, hyper-parameters, etc.
* See https://scikit-learn.org/stable/modules/classes.html for more information.

"""


def _download_dataset(size=10000):
    assert sys.version_info.major == 3, "Use Python3"

    import ssl
    import urllib.request
    url = "https://raw.githubusercontent.com/dongkwan-kim/small_dataset/master/review_{}k.csv".format(size // 1000)

    dir_path = "../data"
    file_path = os.path.join(dir_path, "review_{}k.csv".format(size // 1000))
    if not os.path.isfile(file_path):
        os.makedirs(dir_path, exist_ok=True)
        ctx = ssl._create_unverified_context()
        with urllib.request.urlopen(url, context=ctx) as u, open(file_path, 'wb') as f:
            f.write(u.read())
        print("Download: {}".format(file_path))
    else:
        print("Already exist: {}".format(file_path))


def _get_review_data(path, num_samples, train_test_ratio=0.8):
    """Do not modify the code in this function."""
    _download_dataset()
    print("Load Data at {}".format(path))
    reviews, sentiments = [], []
    with open(path, "r", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for line in reader:
            reviews.append(line["review"])
            sentiments.append(int(line["sentiment"]))

    # Data shuffle
    random.seed(42)
    zipped = list(zip(reviews, sentiments))
    random.shuffle(zipped)
    reviews, sentiments = zip(*(zipped[:num_samples]))
    reviews, sentiments = np.asarray(reviews), np.asarray(sentiments)

    # Train/test split
    num_data, num_train = len(sentiments), int(len(sentiments) * train_test_ratio)
    return (reviews[:num_train], sentiments[:num_train]), (reviews[num_train:], sentiments[num_train:])


def _get_example_of_errors(texts_to_analyze, preds_to_analyze, labels_to_analyze):
    texts_to_analyze = texts_to_analyze[np.random.permutation(len(texts_to_analyze))]
    correct = texts_to_analyze[preds_to_analyze == labels_to_analyze]
    wrong = texts_to_analyze[preds_to_analyze != labels_to_analyze]
    print("\n[Correct Sample Examples]")
    for line in correct[:5]:
        print("\t- {}".format(line))
    print("\n[Wrong Sample Examples]")
    for line in wrong[:5]:
        print("\t- {}".format(line))


def preprocess_and_split_to_tokens(sentences):
    """
    :param sentences: (array_like) array_like objects of strings.
        e.g., ["I like apples", "I love python3"]
    You can choose the level of pre-processing by yourself.
    The easiest way to start is lowering the case (str.lower).

    :return: array_like objects of array_like objects of tokens.
        e.g., [["I", "like", "apples"], ["I", "love", "python3"]]
    """
    punct_mapping = {"_":" ", "'":" "}
    punct = "/-'?!.,#$%\'()*+-/:;<=>@[\\]^_`{|}~" + '""“”’'
    def clean_special_chars(text, punct, mapping):
        for p in punct:
            text = text.replace(p, f' {p} ')
        for m in mapping:
            text = text.replace(m, mapping[m])
        return text

    tokenized_sentences = []

    for sentence in sentences:
        sentence = sentence.lower()
        modified_sentence = clean_special_chars(sentence,punct,punct_mapping)
        tokenized_sentence = modified_sentence.split()
        tokenized_sentences.append(tokenized_sentence)

    return tokenized_sentences
    # raise NotImplementedError


def create_bow(sentences, vocab=None, msg_prefix="\n"):
    """Make the Bag-of-Words model from the sentences, return (vocab, bow_array)
        vocab: dictionary of (token, index of BoW representation) pair.
        bow_array: array_like objects of BoW representation, the shape of which is [#sentence_list, #vocab]

    :param sentences: (array_like): array_like objects of strings
        e.g., ["I like apples", "I love python3"]
    :param vocab: (dict, optional)
        e.g., {"I": 0, "like": 1, "apples": 2, "love": 3, "python3": 4}
    :param msg_prefix: (str)
    :return: Tuple[dict, array_like]
        e.g., ({"I": 0, "like": 1, "apples": 2, "love": 3, "python3": 4},
                [[1, 1, 1, 0, 0], [1, 0, 0, 1, 1]])
    """

    tokens_per_sentence = preprocess_and_split_to_tokens(sentences)
    my_vocab = dict()

    if vocab is None:
        print("{} Vocab construction".format(msg_prefix))
        for sentence in tokens_per_sentence:
            for single_vocab in sentence:
                if single_vocab in my_vocab.keys():
                    continue
                if not my_vocab: # if my_vocab is empty
                    my_vocab[single_vocab] = 0
                else:
                    max_value = max(my_vocab.values())
                    max_value += 1
                    my_vocab[single_vocab] = max_value
        # raise NotImplementedError
    print("{} Bow construction".format(msg_prefix))
    
    tfidf_transformer = TfidfTransformer()
    


    number_of_sentences = len(tokens_per_sentence)

    if vocab is None:
        number_of_vocab = len(my_vocab)
        array_like = [[0 for i in range(number_of_vocab)] for j in range(number_of_sentences)]
        # print("Vocab None number_of_vocab")
        # print(number_of_vocab)
        for i in range(number_of_sentences):
            sentence = tokens_per_sentence[i]
            for single_vocab in sentence:
                if single_vocab not in my_vocab.keys():
                    continue
                vocab_value = my_vocab[single_vocab]
                array_like[i][vocab_value] = 1
        array_like = tfidf_transformer.fit_transform(array_like).toarray()
        return (my_vocab, array_like)
    else:

        number_of_vocab = len(vocab)
        # print("Vocab Not None number_of_vocab")
        # print(number_of_vocab)
        array_like = [[0 for i in range(number_of_vocab)] for j in range(number_of_sentences)]

        for i in range(number_of_sentences):
            sentence = tokens_per_sentence[i]
            for single_vocab in sentence:
                if single_vocab not in vocab.keys():
                    continue
                vocab_value = vocab[single_vocab]
                array_like[i][vocab_value] = 1
        
        array_like = tfidf_transformer.fit_transform(array_like).toarray()

        return (vocab, array_like)
        


def run(test_xs=None, test_ys=None, num_samples=10000, verbose=True):
    """You do not have to consider test_xs and test_ys, since they will be used for grading only."""

    # Data
    (train_xs, train_ys), (val_xs, val_ys) = _get_review_data(path="../data/review_10k.csv", num_samples=num_samples)
    
    if verbose:
        print("\n[Example of xs]: [\"{}...\", \"{}...\", ...]\n[Example of ys]: [{}, {}, ...]".format(
            train_xs[0][:70], train_xs[1][:70], train_ys[0], train_ys[1]))
        print("\n[Num Train]: {}\n[Num Test]: {}".format(len(train_ys), len(val_ys)))

    # Create bow representation of train set
    my_vocab, train_bows = create_bow(train_xs, msg_prefix="\n[Train]")
    assert isinstance(my_vocab, dict)
    assert isinstance(train_bows, list) or isinstance(train_bows, np.ndarray) or isinstance(train_bows, tuple)
    if verbose:
        print("\n[Vocab]: {} words".format(len(my_vocab)))

    # You can see hyper-parameters (train_kwargs) that can be tuned in the document below.
    #   https://scikit-learn.org/stable/modules/classes.html.
    train_kwargs = dict(verbose=1, solver="liblinear")
    # clf = LogisticRegression(**train_kwargs)
    # clf = RandomForestClassifier(n_estimators=800, max_depth=64)
    # clf = svm.LinearSVC()
    clf = svm.SVC(gamma='auto')
    clf.fit(train_bows, train_ys)
    assert hasattr(clf, "predict")

    # Create bow representation of validation set
    _, val_bows = create_bow(val_xs, vocab=my_vocab, msg_prefix="\n[Validation]")

    # Evaluation
    val_preds = clf.predict(val_bows)
    val_accuracy = accuracy_score(val_ys, val_preds)
    if verbose:
        print("\n[Validation] Accuracy: {}".format(val_accuracy))
        _get_example_of_errors(val_xs, val_preds, val_ys)

    # n = [100,200,400,800]
    # depth = [16,32,64]
    # alpha_list = [0.8]
    # max_val_accuracy = 0
    # for alpha in alpha_list:
        
    #     clf = MultinomialNB(alpha=alpha)
    #     clf.fit(train_bows, train_ys)
    #     assert hasattr(clf, "predict")

    #     # Create bow representation of validation set
    #     _, val_bows = create_bow(val_xs, vocab=my_vocab, msg_prefix="\n[Validation]")

    #     # Evaluation
    #     val_preds = clf.predict(val_bows)
    #     val_accuracy = accuracy_score(val_ys, val_preds)
    #     if max_val_accuracy < val_accuracy:
    #         # print("n_estimator : %d max_depth : %d" %(n_estimator, max_depth))
    #         print("alpha : ",alpha)
    #         print("\n[Validation] Accuracy: {}".format(val_accuracy))
    #     # if verbose:
    #     #     print("\n[Validation] Accuracy: {}".format(val_accuracy))
    #     #     _get_example_of_errors(val_xs, val_preds, val_ys)

    # Grading: Do not modify below lines.
    if test_xs is not None:
        _, test_bows = create_bow(test_xs, vocab=my_vocab, msg_prefix="\n[Test]")
        test_preds = clf.predict(test_bows)
        return {"clf": clf, "val_accuracy": val_accuracy, "test_accuracy": accuracy_score(test_ys, test_preds)}
    else:
        return {"clf": clf}


if __name__ == '__main__':
    run()
