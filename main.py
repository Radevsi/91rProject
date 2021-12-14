# Initialize the globals

from matplotlib.colors import from_levels_and_colors


EPOCHS = 1 
LEARNING_RATE = 2e-3 
HIDDEN_SIZE = 64
LAYERS = 3  
VERBOSE = True
FRACTIONS = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] 


def parse_args():
    from sys import argv
    from utils.load_data import load_datasets, load_words2num, load_vi2en

    n_correct_args = 1

    # defaults
    data_dir = 'words2num'
    run_rnn_model = False # default run the transformer
    run_make_metrics = False
    run_make_metrics_all = False
    save = True # should we save anything
    save_ow = False # if the same model has been trained, train it again and re-save it

    # Tells us how to run the model
    if '--mt' in argv: # run the machine translation task
        data_dir = 'vi2en'
        n_correct_args += 1
    if '--rnn' in argv: # run the rnn-based encoder-decoder model
        run_rnn_model = True
        n_correct_args += 1
    if '--make-metrics' in argv: # get the metrics for experiments
        run_make_metrics = True
        n_correct_args += 1
    if '--make-metrics-all' in argv: # get the metrics on all available models
        run_make_metrics = True
        run_make_metrics_all = True
        n_correct_args += 1
    if '--no-save' in argv: # don't save any models or results/parameters
        print("Warning: this run will not save any models or values")
        save = False
        n_correct_args += 1
    if '--save-ow' in argv:
        print("Warning: this run will overwrite any matching models saved from previous runs")
        save_ow = True
        n_correct_args += 1

    # Check for parameter settings (like number of epochs, learning rate, etc.) 
    for arg in argv:
        if 'epochs=' in arg:
            global EPOCHS
            EPOCHS = arg[7:]
            n_correct_args += 1
        if 'lr=' in arg:
            global LEARNING_RATE
            LEARNING_RATE = arg[3:]
            n_correct_args += 1
        if 'hidden=' in arg:
            global HIDDEN_SIZE
            HIDDEN_SIZE = arg[7:]
            n_correct_args += 1
        if 'layers=' in arg:
            global LAYERS
            LAYERS = arg[7:]
            n_correct_args += 1
        if '--verbose-' in arg:
            global VERBOSE
            try: VERBOSE = bool(int(arg[10:]))
            except ValueError: 
                print("Set verbose argument to 0 or 1")
                exit(1)
            n_correct_args += 1
        if '--frac-' in arg:
            start = arg[7:8]
            assert arg[8:9] == '-', "Improper command line argument usage for --frac"
            end = arg[9:]
            global FRACTIONS
            try:
                start, end = int(start), int(end)
            except ValueError:
                print("Parameters passed to --frac- must be ints")
                exit(1)
            FRACTIONS = [i*0.1 for i in range(start, end+1)]
            n_correct_args += 1
        try:
            EPOCHS, LEARNING_RATE = int(EPOCHS), float(LEARNING_RATE)
            HIDDEN_SIZE, LAYERS = int(HIDDEN_SIZE), int(LAYERS)
        except ValueError:
            print("Parameter settings must be numbers of an appropriate type.")
            exit()

    # Just load the data
    if '--load-data' in argv: # load all the data then exit
        load_datasets()
        exit()
    elif '--load-words2num' in argv: # load the words2num dataset
        load_words2num()
        exit()
    elif '--load-vi2en' in argv: # load the vi2en dataset
        load_vi2en()
        exit()
    elif '-h' in argv or '--help' in argv:
        print("Usage: python main.py [args]\n"
              "Supported [args]:\n"
              "  --mt: Run the machine translation task on the vi2en dataset\n"
              "  --rnn: Run the rnn-based encoder-decoder model\n"
              "  --make-metrics: Run the selected model on splits of training data and return metrics\n"
              "  --make-metrics-all: Run all models on splits of training data and return metrics\n"
              "  --no-save: Don't save any models or results/parameters\n"
              "  --save-ow: Overwrite save. Overwrite any matching, previously saved models/results\n"
              "  --load-data: Load all the datasets and exit without processing\n"
              "  --load-words2num: Load the words2num dataset and exit without processing\n"
              "  --load-vi2en: Load the vi2en dataset and exit without processing\n"
              "  --verbose-[x]: Print verbose output about data if x is true\n"
              "  --frac-[x]-[y]: Set the endpoints of the data splitting. x and y must be ints.\n"
              "  epochs=[x]: Set number of epochs to x (must be an int)\n"
              "  lr=[x]: Set learning rate to x (must be a float)\n"
              "  hidden=[x]: Set hidden size to x (must be an int)\n"
              "  layers=[x]: Set number layers to x (must be an int)\n" 
              "  --help or -h: Run this list of commands\n"
              "By default, main.py will run the transformer model on the words2num dataset."
        )     
        exit()
    elif len(argv) > n_correct_args:
        print(f"Undefined argument present.", end=" ")
        print("Run with --help or -h for a list of supported arguments.")
        exit()                 
    return data_dir, run_rnn_model, run_make_metrics, run_make_metrics_all, save, save_ow

def main():

    # First parse the command-line
    data_dir, run_rnn, run_make_metrics, run_make_metrics_all, save, save_ow = parse_args()

    # Now make the imports
    import torch
    from collections import defaultdict
    from utils.process_data import process_data
    from models.transformer import TransformerEncoderDecoder
    from models.Bi_LSTM import AttnEncoderDecoder    
    from utils.utils import run_pipeline, make_metrics, plot_metrics

    # GPU check
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print (device)

    # Loading is handled in processing        
    SRC, TGT, train_iter, val_iter, test_iter = process_data(data_dir, verbose=VERBOSE)

    # Instantiate and train appropriate classifier (transformer by default)
    attn_model = AttnEncoderDecoder(SRC, TGT,
        hidden_size    = HIDDEN_SIZE,
        layers         = LAYERS,
    ).to(device)
    transformer_model = TransformerEncoderDecoder(SRC, TGT,
        hidden_size    = HIDDEN_SIZE,
        layers         = LAYERS,
    ).to(device)
    models = [attn_model, transformer_model]

    if run_rnn:
        model = models[0] # AttnEncoderDecoder
    else:
        model = models[1] # TransformerEncoderDecoder

    # Train the model or run experiments (by default just train)
    if run_make_metrics:
        # Get splits of the batches of the training data (if necessary)
        # fractions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        # fractions = [0.001, 0.002]        
        if run_make_metrics_all:
            metrics, total_times = make_metrics(models, train_iter, val_iter, test_iter, data_dir, FRACTIONS,
                EPOCHS, LEARNING_RATE, HIDDEN_SIZE, LAYERS, save, save_ow)
        else:
            metrics, total_times = make_metrics([model], train_iter, val_iter, test_iter, data_dir, FRACTIONS,
                EPOCHS, LEARNING_RATE, HIDDEN_SIZE, LAYERS, save, save_ow)

        heading = "Models Trained On {} Epochs, {} Learning Rate, {} Hidden Size, and {} Layers".\
                    format(EPOCHS, LEARNING_RATE, HIDDEN_SIZE, LAYERS)

        png_name = f"{EPOCHS}es_{LEARNING_RATE}lr_{HIDDEN_SIZE}h_{LAYERS}l" 

        plot_metrics(metrics, total_times, heading, png_name)

    else:
        run_pipeline(model, 
                     train_iter, 
                     val_iter, 
                     test_iter, 
                     data_dir, 
                     epochs=EPOCHS, 
                     lr=LEARNING_RATE, 
                     hz=HIDDEN_SIZE, 
                     layers=LAYERS, 
                     save=save, 
                     save_ow=save_ow
                    )

if __name__ == '__main__': 
    main()
