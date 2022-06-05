import logging
from abc import ABC

import pandas as pd
import torch
from torch.utils.data import DataLoader, RandomSampler, SequentialSampler, random_split
from transformers import GPT2LMHeadModel, GPT2Config
from transformers import get_linear_schedule_with_warmup

import src.config as CONF
import src.constants as CONST
import src.schema as S
import src.utils as U
from src.core import BaseTransformer
from src.loading_dataset import DatasetLoader

torch.manual_seed(42)

logging.basicConfig(format='%(asctime)s:%(levelname)s:%(message)s')
log = logging.getLogger('pipeline-log')
log.setLevel(logging.DEBUG)


class PipelineJokeGeneration(ABC):
    def __init__(self,
                 preprocessor: BaseTransformer,
                 tokenizer: object,
                 model_name: str,
                 splitting_params: dict = None,
                 ):
        """
        Pipeline to unite data processing, model training and generating text

        :param preprocessor: basic preprocessor instance
        :param tokenizer: basic text tokenizer
        :param model_name: custom model
        :param splitting_params: dict of parameters for splitting (see split_df)
        """
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.model_name = model_name
        self.splitting_params = splitting_params
        self.model = None
        self.optimizer = None

    # def run_joke_generator(
    #         self,
    #         df: pd.DataFrame,
    #         learning_rate,
    #         optimizer
    # ):
    #     """
    #     Run pipeline
    #     :param df: dataframe
    #     :return recommendations: dictionary of top N catalog items for each user_id
    #     :return scores: scores for model performance
    #     """
    #     df = df.copy()
    #     log.info('Running joke generation pipeline')
    #     train_dataloader, validation_dataloader = self.prepare_data(df)
    #     # self.train_model(X_train, y_train)
    #     # y_pred = self.get_prediction(X_test)
    #     #
    #     # log.info('Counting scores')
    #     # scores = self.get_scores(np.array(y_test), np.array(y_pred))
    #     # return scores
    #     pass

    def train_model(self,
                    learning_rate,
                    optimizer: object,
                    train_dataloader,
                    validation_dataloader,
                    epochs=CONST.EPOCHS,
                    device=CONF.DEVICE
                    ):
        """
        Train model
        """
        log.info('Training model')
        model = None
        if self.model_name == 'GPT2':
            # Loading the model configuration and setting it to the GPT2 standard settings.
            configuration = GPT2Config.from_pretrained('gpt2', output_hidden_states=False)
            # Create the instance of the model and set the token size embedding length
            model = GPT2LMHeadModel.from_pretrained("gpt2", config=configuration)
            model.resize_token_embeddings(len(self.tokenizer))
            model.to(CONF.DEVICE)  # cuda()

        U.seed_everything(CONF.seed_val)
        optimizer = optimizer(model.parameters(), lr=learning_rate, eps=1e-8)

        self.model = model
        self.optimizer = optimizer

        """
        Total training steps is the number of data points, times the number of epochs. 
        Essentially, epochs are training cycles, how many times each point will be seen by the model. 
        """

        total_steps = len(train_dataloader) * epochs

        """
        We can set a variable learning rate which will help scan larger areas of the 
        problem space at higher LR earlier, then fine tune to find the exact model minima 
        at lower LR later in training.
        """
        scheduler = get_linear_schedule_with_warmup(optimizer,
                                                    num_warmup_steps=CONF.warmup_steps,
                                                    num_training_steps=total_steps)

        training_stats = U.train_generation(self.model, train_dataloader,
                                            validation_dataloader, self.optimizer,
                                            self.tokenizer, scheduler,
                                            epochs, device)
        return training_stats

    def draw_curves(self,
                    training_stats
                    ):
        """
        Visualize loss  of the model
        """
        U.draw_train_val_loss(training_stats)

    def generate(self,
                 ):
        log.info('Starting jokes generation')
        self.model.eval()
        U.seed_everything(42)

        prompt = "<|startoftext|>"

        generated = torch.tensor(self.tokenizer.encode(prompt)).unsqueeze(0)
        generated = generated.to(CONF.DEVICE)

        sample_outputs = self.model.generate(
            generated,
            do_sample=True,
            top_k=50,
            max_length=300,
            top_p=0.95,
            num_return_sequences=25
        )

        for i, sample_output in enumerate(sample_outputs):
            print("{}: {}\n\n".format(i, self.tokenizer.decode(sample_output, skip_special_tokens=True)))

    def save_model(self, output_dir):
        """
        Save a trained model, configuration and tokenizer using `save_pretrained()`.
        """
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(output_dir)
        self.tokenizer.save_pretrained(output_dir)

    def _tokenize(self,
                  df: pd.DataFrame
                  ):
        joke_tokens = pd.DataFrame(df[[S.JOKE]].copy())
        joke_tokens["len_tokens"] = joke_tokens.applymap(lambda x: len(self.tokenizer.encode(x)))
        joke_tokens = joke_tokens.sort_values("len_tokens", ascending=False)
        max_n_tokens = joke_tokens["len_tokens"].max()
        jokes = joke_tokens.joke[joke_tokens.len_tokens <= max_n_tokens]
        return jokes, max_n_tokens

    def prepare_data(self,
                     df: pd.DataFrame,
                     batch_size=CONST.BATCH_SIZE_GEN,
                     ):
        """
        Prepare data for training and text generation
        :param df: Dataframe to prepare
        :param batch_size: size of one batch
        :return: train_dataloader, validation_dataloader
        """
        df = df.copy()

        log.debug('Starting preprocessing')
        self.preprocessor.fit(df)
        df = self.preprocessor.transform(df)

        # get just best jokes
        mean_len = int(df[(df[S.TARGET] == S.HIGHEST_SCORE)][S.TXT_WORD_CNT].mean())
        df = df[
            (df[S.TARGET] == S.HIGHEST_SCORE) & (df[S.TXT_WORD_CNT] >= S.MIN_WORDS) & (df[S.TXT_WORD_CNT] <= mean_len)]
        df.dropna(inplace=True)

        log.debug('Starting tokenization')
        best_jokes, max_n_tokens = self._tokenize(df)
        dataset = DatasetLoader.JokeDataset(best_jokes, self.tokenizer, max_length=max_n_tokens)

        log.debug('Splitting dataset on train and val datasets')
        train_size = int(0.9 * len(dataset))
        val_size = len(dataset) - train_size
        train_df, val_df = random_split(dataset, [train_size, val_size])

        train_dataloader = DataLoader(
            train_df,
            sampler=RandomSampler(train_df),
            batch_size=batch_size
        )
        validation_dataloader = DataLoader(
            val_df,
            sampler=SequentialSampler(val_df),
            batch_size=batch_size
        )

        return train_dataloader, validation_dataloader
