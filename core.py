import argparse
import inspect
from keras import losses, optimizers
from keras.models import load_model
from lee_dernoncourt import lee_dernoncourt
from embedding import read_word2vec, read_glove_twitter
from dataset import load_swda_corpus_data

models =     {
                'Lee-Dernoncourt': lee_dernoncourt,
                'KADJK': kadjk
             }

embeddings = {
                'word2vec': read_word2vec,
                'GloVe': read_glove_twitter
             }

datasets =   {
                'SwDA': load_swda_corpus_data
             }


def print_options(option_dict):
    for key in option_dict.keys():
        print('\t' + key)

def check_keras_option_validity(option_given, keras_option_data):
    for a, b in keras_option_data:
        if a == option_given:
            return True
    return False


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Short-text classification training tool.')
    parser.add_argument('--loss-functions', action='store_true', help='Print available loss functions.')
    parser.add_argument('--optimizers', action='store_true', help='Print available optimizers.')
    parser.add_argument('--models', action='store_true', help='Print available models.')
    parser.add_argument('--embeddings', action='store_true', help='Print possible word embeddings.')
    parser.add_argument('--datasets', action='store_true', help='Print possible datasets.')

    parser.add_argument('--model', type=str, help='Model to use.')
    parser.add_argument('--dataset', nargs=2, metavar=('TYPE', 'PATH'), type=str, help='Dataset to use.')
    parser.add_argument('--embedding', nargs=2, metavar=('TYPE', 'PATH'), type=str, help='Embedding to use.')
    parser.add_argument('--loss-function', type=str, help='Loss function to use.')
    parser.add_argument('--optimizer', type=str, help='Optimizer to use.')
    parser.add_argument('--save-model', action='store_true', help='Save model once training is complete.')
    parser.add_argument('--load-model', type=str, help='Load pretrained model from a .h5 file and print its accuracy.')

    parser.add_argument('--train', nargs=1, metavar=('NUM_EPOCHS'), type=int, help='Train the specified network for given number of epochs.')
    # TODO: Add a parameter that helps a trained network evaluate a sample conversation

    args = parser.parse_args()
    if args.loss_functions:
        loss_functions = inspect.getmembers(losses, inspect.isfunction)
        print('Loss functions available:')
        for (a, b) in loss_functions:
            print('\t' + a)
    elif args.optimizers:
        optimizer_classes = inspect.getmembers(optimizers, inspect.isclass)
        print('Optimizers available:')
        for (a, b) in optimizer_classes:
            print('\t' + a)
    elif args.models:
        print('Models available:')
        print_options(models)
    elif args.embeddings:
        print('Embeddings available:')
        print_options(embeddings)
    elif args.datasets:
        print('Datasets available:')
        print_options(datasets)
    else:
        if args.loss_function and args.optimizer and\
           args.model and args.embedding and args.dataset and\
           models[args.model] and embeddings[args.embedding[0]] and datasets[args.dataset[0]]:
            loss_valid = check_keras_option_validity(args.loss_function,
                                                     inspect.getmembers(losses, inspect.isfunction))
            optimizer_valid = check_keras_option_validity(args.optimizer,
                                                          inspect.getmembers(optimizers, inspect.isclass))
            if loss_valid and optimizer_valid:
                loss_function = args.loss_function
                optimizer = args.optimizer
                model = models[args.model]
                embedding_loading_function = embeddings[args.embedding[0]]
                embedding_file_path = args.embedding[1]
                dataset_loading_function = datasets[args.dataset[0]]
                dataset_file_path = args.dataset[1]
                load_from_model_file = args.load_model
                save_model = (args.save_model is not None)

                if args.train:
                    num_epochs_to_train = args.train[0]
                else:
                    num_epochs_to_train = 0

                model_filename = args.model + '_' + args.embedding[0] + '_' +\
                                 args.dataset[0] + '_' + args.loss_function + '_' +\
                                 args.optimizer + '_' + str(num_epochs_to_train) + '.h5'

                model(dataset_loading_function, dataset_file_path,
                      embedding_loading_function, embedding_file_path,
                      num_epochs_to_train, loss_function, optimizer,
                      load_from_model_file, save_model, model_filename)
        else:
            print("Please enter all required argument. Use --help to review required arguments.")