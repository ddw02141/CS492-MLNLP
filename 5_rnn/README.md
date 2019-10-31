# RNN

In this task, you will implement a RNN model for binary sentiment classification.

Implement three methods:
- encode_text(sentences, vectorizer, max_len, msg_prefix, verbose) -> vectorizer, max_len, pad_encoded_sentences
- build_model(learning_rate, max_len, num_classes, num_vocab, num_embed, num_hidden, num_lstm_cells, l2_lambda)
    -> X, Y, keep_prob, optimizer, cost, predictions
- train_model(session, X, Y, keep_prob, optimizer, cost, predictions,
              train_xs, train_ys, val_xs, val_ys, batch_size, total_epoch, keep_prob_value, verbose) -> None

## Instruction

* See skeleton codes below for more details.
* Do not remove assert lines and do not modify methods that start with an underscore.

## Important Notes
* TF 2.0 has been released recently, but our task is designed with TF 1.4
* Our code is compatible to 1.5 which is the default version of current Google Colab.
* We highly recommend using Google Colab with GPU for students who have not a GPU in your local or remote computer.
    - Runtime > Change runtime type > Hardware accelerator: GPU
* For one epoch of training, Colab+GPU takes 35s, Colab+CPU takes 165s, and local laptop (TA's) takes 140s.
* TA's code got 82-84 validation accuracy in a total of 545-555s (15 epochs) at Colab+GPU.