from bs4 import BeautifulSoup
import re
import unidecode
import contractions
from collections import Counter
import os
import torch
import numpy as np
import src.schema as S
import tqdm
import matplotlib.pyplot as plt
import src.config as CONF
import random
import time
import datetime
import seaborn as sns
import pandas as pd


# Removal of html tags
def strip_html_tags(text):
    """remove html tags from text"""
    soup = BeautifulSoup(text, "html.parser")
    stripped_text = soup.get_text(separator=" ")
    return stripped_text


# Removal of whitespaces
def remove_whitespace(text):
    """remove extra whitespaces from text"""
    text = text.strip()
    return " ".join(text.split())


# Removal of accented characters
def remove_accented_chars(text):
    """remove accented characters from text, e.g. café"""
    text = unidecode.unidecode(text)
    return text


# Removal of shortened words
def expand_contractions(text):
    """expand shortened words, e.g. don't to do not"""
    text = contractions.fix(text)
    return text


# Removal of urls
def find_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.findall(text)


def remove_urls(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    return url_pattern.sub(r'', text)


# Removal of Frequent words
def count_words(text, top=10):
    cnt = Counter()
    for text in text.values:
        for word in text.split():
            cnt[word] += 1

    return cnt.most_common(top)


# Removal of numbers
def remove_numbers(inp):
    input_str = re.sub(r'\d+', "", inp)
    return input_str


# Save model or checkpoint
def save_checkpoint(model, optimizer, learning_rate, iteration, filepath):
    print("Saving model and optimizer state at iteration {} to {}".format(
        iteration, filepath))

    model_state_dict = model.state_dict().copy()

    torch.save({'iteration': iteration,
                'state_dict': model_state_dict,
                'optimizer': optimizer.state_dict(),
                'learning_rate': learning_rate}, filepath)


# Load model or checkpoint
def load_checkpoint(checkpoint_path, model, optimizer):
    assert os.path.isfile(checkpoint_path)
    print("Loading checkpoint '{}'".format(checkpoint_path))
    checkpoint_dict = torch.load(checkpoint_path, map_location='cpu')

    model.load_state_dict(checkpoint_dict['state_dict'])
    optimizer.load_state_dict(checkpoint_dict['optimizer'])
    learning_rate = checkpoint_dict['learning_rate']
    iteration = checkpoint_dict['iteration']
    print("Loaded checkpoint '{}' from iteration {}".format(
        checkpoint_path, iteration))
    return model, optimizer, learning_rate, iteration


#########################################
def as_matrix(convertor, sequences, max_len=20):
    """ Convert a list of tokens into a matrix with padding """
    if isinstance(sequences[0], str):
        sequences = list(map(str.split, sequences))

    max_len = min(max(map(len, sequences)), max_len or float('inf'))

    UNK_IX, PAD_IX = convertor.get_unk_pad_ix()

    matrix = np.full((len(sequences), max_len), np.int32(PAD_IX))
    for i, seq in enumerate(sequences):
        token_to_id = convertor.get_token_to_id()
        row_ix = [token_to_id.get(word, UNK_IX) for word in seq[:max_len]]
        matrix[i, :len(row_ix)] = row_ix

    return matrix


def to_tensors(batch, device):
    batch_tensors = dict()
    for key, arr in batch.items():
        if key in [S.JOKE]:
            batch_tensors[key] = torch.tensor(arr, device=device, dtype=torch.int64)
        else:
            batch_tensors[key] = torch.tensor(arr, device=device)  # dtype=torch.float64
    return batch_tensors


def make_batch(convertor, data, max_len=None,
               word_dropout=0, device=torch.device('cpu')):
    """
    Creates a keras-friendly dict from the batch data.
    :param word_dropout: replaces token index with UNK_IX with this probability
    :returns: a dict with {'title' : int64[batch, title_max_len]
    """
    batch = {S.JOKE: as_matrix(convertor, data[S.JOKE].values, max_len)}

    if word_dropout != 0:
        batch[S.JOKE] = apply_word_dropout(batch[S.JOKE], 1. - word_dropout)

    if S.TARGET in data.columns:
        batch[S.TARGET] = data[S.TARGET].values

    return to_tensors(batch, device)


def apply_word_dropout(matrix, keep_prop,
                       replace_with,
                       pad_ix, ):
    dropout_mask = np.random.choice(2, np.shape(matrix), p=[keep_prop, 1 - keep_prop])
    dropout_mask &= matrix != pad_ix
    return np.choose(dropout_mask, [matrix, np.full_like(matrix, replace_with)])


def iterate_minibatches(convertor, data, batch_size=256,
                        shuffle=True, cycle=False, **kwargs):
    """ iterates minibatches of data in random order """
    while True:
        indices = np.arange(len(data))
        if shuffle:
            indices = np.random.permutation(indices)

        for start in range(0, len(indices), batch_size):
            batch = make_batch(convertor, data.iloc[indices[start: start + batch_size]], **kwargs)
            yield batch

        if not cycle: break


def train(convertor, model,
          optimizer, criterion,
          data_train, data_val,
          batch_size=128,
          epochs=100, iter_per_validation=50,
          early_stopping=False,
          checkpoint_path="./best_checkpoint",
          save_by="accuracy",
          device=CONF.DEVICE, **kw):
    count = 0
    patience = 5

    iteration = 0
    loss_list = []
    iteration_list = []
    accuracy_list = []

    best_score = None

    for epoch in range(epochs):
        print(f"Epoch {epoch}")

        # for vec, vlen, labels in train_loader:
        for i, batch in tqdm.notebook.tqdm(
                enumerate(iterate_minibatches(
                    convertor,
                    data_train,
                    batch_size=batch_size,
                    device=CONF.DEVICE)),
                total=len(data_train) // batch_size
        ):
            model.train()
            # Clear gradients
            optimizer.zero_grad()

            # Forward propagation
            outputs = model(batch)

            # Calculate softmax and ross entropy loss
            loss = criterion(outputs, batch[S.TARGET])

            # Calculating gradients
            loss.backward()

            # Update parameters
            optimizer.step()

            iteration += 1

            if iteration % iter_per_validation == 99:
                model.eval()
                with torch.no_grad():
                    # Calculate Accuracy
                    correct = 0
                    total = 0

                    for i, batch in enumerate(iterate_minibatches(
                            convertor,
                            data_val,
                            batch_size=batch_size,
                            device=CONF.DEVICE)):
                        # Forward propagation
                        outputs = model(batch)

                        # Get predictions from the maximum value
                        predicted = torch.max(outputs.data, 1)[1]

                        # Total number of labels
                        total += len(batch[S.TARGET])

                        correct += (predicted == batch[S.TARGET]).sum()

                    accuracy = 100 * correct / float(total)

                    # store loss and iteration
                    loss_list.append(loss.data.detach().cpu().numpy())
                    iteration_list.append(iteration)
                    accuracy_list.append(accuracy.detach().cpu().numpy())
                    print('Iteration: {}  Loss: {}  Accuracy: {} %'.format(iteration, loss.data, accuracy))

                    # save if we have best model state
                    ref_score = accuracy if save_by == "accuracy" else loss.data
                    compare = (lambda x, y: x > y) if save_by == "accuracy" else (lambda x, y: x < y)
                    if best_score is None or compare(ref_score, best_score):
                        best_score = ref_score
                        save_checkpoint(model, optimizer,
                                        optimizer.state_dict()['param_groups'][0]['lr'],
                                        iteration, checkpoint_path)

                    # Early stopping if the current valid_loss is greater than the last three valid losses
                    if early_stopping == True:
                        if len(accuracy_list) > 3 and all(accuracy >= acc for acc in accuracy_list[-4:]):
                            print('Stopping early')
                            break

    return iteration_list, loss_list, accuracy_list


def draw_visualization(iteration_list, loss_list, accuracy_list):
    # visualization loss
    plt.plot(iteration_list, loss_list)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.title("Loss vs Number of iteration")
    plt.show()

    # visualization accuracy
    plt.plot(iteration_list, accuracy_list, color="red")
    plt.xlabel("Number of iteration")
    plt.ylabel("Accuracy")
    plt.title("Accuracy vs Number of iteration")
    plt.show()


# reproducibility
def seed_everything(seed_val):
    random.seed(seed_val)
    np.random.seed(seed_val)
    torch.manual_seed(seed_val)
    torch.cuda.manual_seed_all(seed_val)


# train generation
def format_time(elapsed):
    return str(datetime.timedelta(seconds=int(round((elapsed)))))


def train_generation(model,
                     train_dataloader,
                     validation_dataloader,
                     optimizer,
                     tokenizer,
                     scheduler,
                     epochs: int,
                     device):
    total_t0 = time.time()

    training_stats = []

    model = model.to(device)

    for epoch_i in range(0, epochs):

        print(f'Beginning epoch {epoch_i + 1} of {epochs}')

        t0 = time.time()

        total_train_loss = 0

        model.train()

        for step, batch in enumerate(train_dataloader):

            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            model.zero_grad()

            outputs = model(b_input_ids,
                            labels=b_labels,
                            attention_mask=b_masks,
                            token_type_ids=None
                            )

            loss = outputs[0]

            batch_loss = loss.item()
            total_train_loss += batch_loss

            # Get sample every 100 batches.
            if step % CONF.sample_every == 0 and not step == 0:

                elapsed = format_time(time.time() - t0)
                print(f'Batch {step} of {len(train_dataloader)}. Loss:{batch_loss}. Time:{elapsed}')

                model.eval()

                sample_outputs = model.generate(
                    bos_token_id=random.randint(1, 30000),
                    do_sample=True,
                    top_k=50,
                    max_length=200,
                    top_p=0.95,
                    num_return_sequences=1
                )
                for i, sample_output in enumerate(sample_outputs):
                    print(f'Example output: {tokenizer.decode(sample_output, skip_special_tokens=True)}')

                model.train()

            loss.backward()

            optimizer.step()

            scheduler.step()

        # Calculate the average loss over all of the batches.
        avg_train_loss = total_train_loss / len(train_dataloader)

        # Measure how long this epoch took.
        training_time = format_time(time.time() - t0)

        print(f'Average Training Loss: {avg_train_loss}. Epoch time: {training_time}')

        t0 = time.time()

        model.eval()

        total_eval_loss = 0
        nb_eval_steps = 0

        # Evaluate data for one epoch
        for batch in validation_dataloader:
            b_input_ids = batch[0].to(device)
            b_labels = batch[0].to(device)
            b_masks = batch[1].to(device)

            with torch.no_grad():
                outputs = model(b_input_ids,
                                attention_mask=b_masks,
                                labels=b_labels)

                loss = outputs[0]

            batch_loss = loss.item()
            total_eval_loss += batch_loss

        avg_val_loss = total_eval_loss / len(validation_dataloader)

        validation_time = format_time(time.time() - t0)

        print(f'Validation loss: {avg_val_loss}. Validation Time: {validation_time}')

        # Record all statistics from this epoch.
        training_stats.append(
            {
                'epoch': epoch_i + 1,
                'Training Loss': avg_train_loss,
                'Valid. Loss': avg_val_loss,
                'Training Time': training_time,
                'Validation Time': validation_time
            }
        )

    print(f'Total training took {format_time(time.time() - total_t0)}')
    return training_stats


def draw_train_val_loss(training_stats):
    pd.set_option('precision', 2)
    df_stats = pd.DataFrame(data=training_stats)
    df_stats = df_stats.set_index('epoch')

    # Use plot styling from seaborn.
    sns.set(style='darkgrid')

    # Increase the plot size and font size.
    sns.set(font_scale=1.5)
    plt.rcParams["figure.figsize"] = (12, 6)

    # Plot the learning curve.
    plt.plot(df_stats['Training Loss'], 'b-o', label="Training")
    plt.plot(df_stats['Valid. Loss'], 'g-o', label="Validation")

    # Label the plot.
    plt.title("Training & Validation Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.xticks([1, 2, 3, 4])

    plt.show()
