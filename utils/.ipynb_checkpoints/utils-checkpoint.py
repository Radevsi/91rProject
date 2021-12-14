# Some useful functions for performing the experiments
from os import listdir
from time import time
from collections import defaultdict
import torch
# from torchsummary import summary

def _split_train(train_iter, fractions, batch_dict=defaultdict(list)):
    """Split the training data into different sizes. 
        Returns a dictionary with a key-value pair of 
        fraction, list of training batches
    """
    n_batches = len(train_iter)
    print("Creating dictionary of batches for {} different sizes".format(len(fractions)))
    start_time = time()
    for fraction in fractions:
        batches = []
        stop_size = int(fraction * n_batches)
        # print("stop size is {}".format(stop_size))
        for i, batch in enumerate(train_iter):
            if i > stop_size:
            # print(i)
                break
            batches.append(batch)
        batch_dict[fraction] = batches

    print("split_train() finished execution in {:.2f} seconds".format(time() - start_time))

    return batch_dict

def run_pipeline(model, train_iter, val_iter, test_iter, data_dir, fraction=1.0,
                 epochs=1, lr=2e-3, hz=64, layers=3, save=True, save_ow=False):
    """
    model: an instantiated model classifier
    train_iter, val_iter, test_iter: data sets
    epochs, lr, hz, layers: model parameters
    save: boolean switch to tell us to save the model or not
    save_ow: overwrite any existing models
    """
    # summary(model, ())
    # Define paths to save to
    PATH_TO_MODELS = 'saved/models/'
    PATH_TO_RESULTS = 'saved/results/'
    file_name = f"{epochs}es_{lr}lr_{hz}h_{layers}l_{fraction}frac.pt" 
    should_train = True
    if save:
        models_path = PATH_TO_MODELS + f'{model.name}/{file_name}'
        results_path = PATH_TO_RESULTS + f'{model.name}/{file_name}'
        if file_name in listdir(PATH_TO_MODELS + '/' + model.name) and not save_ow:
            print(f"Loading model {file_name} found in path...")
            should_train = False
            model.load_state_dict(torch.load(models_path))
            model.eval()
            print("Loading results found in path...")
            results = torch.load(results_path)

    # Training
    if should_train:
        print(f"""Begin training {model.name} with {epochs} epochs, {lr} learning rate,
                  {hz} hidden size and {layers} layers on {fraction:.2%} of {data_dir} dataset.
               """)
        start_time = time()
        val_ppls = model.train_all(train_iter, val_iter, epochs=epochs, learning_rate=lr)
        training_time = time() - start_time
        model.load_state_dict(model.best_model)

        if save:
            print ("Path to save trained model to: {}".format(models_path))
            torch.save(model.state_dict(), models_path) # save model

        # Evaluation 
        print("Evaluating model on testing data...")
        start_time = time()
        test_ppl = model.evaluate_ppl(test_iter)
        eval_time = time() - start_time

        print(f"Testing Perplexity: {test_ppl:.4f}")

        results = test_ppl, val_ppls, training_time, eval_time
        print(f"{model.name} took {training_time:.2f} seconds to train and {eval_time:.2f} seconds to evaluate.")
        if save:
            print("Saving results to file...")    
            torch.save(results, results_path)

    return results

def make_metrics(models, train_iter, val_iter, test_iter, data_dir, fractions,
                 epochs, lr, hz, layers, save, save_ow):
    """
    models: a list of models to evaluate
    The rest of the parameters same as in run_pipeline.
    """

    train_splits = _split_train(train_iter, fractions=fractions)
    print (f"""Evaluating metrics for models {[model.name for model in models]} with {epochs} epochs, 
                {lr} learning rate, {hz} hidden size and 
                {layers} layers on {data_dir} dataset.
            """)

    # Datastructure to store the metrics on all models - value is another dict with model key
    metrics = defaultdict(dict)
    total_times = defaultdict(dict)

    print('making metrics...')
    for fraction in train_splits:
        train_iter = train_splits[fraction]
        for model in models:
            test_ppls, val_ppls, training_time, eval_time = \
                run_pipeline(model, train_iter, val_iter, test_iter, data_dir, fraction,
                             epochs, lr, hz, layers, save, save_ow)
            metrics[model][fraction] = test_ppls, val_ppls
            total_times[model][fraction] = training_time, eval_time
    for model in models:
        print()
        print(f"{model.name} metrics: {metrics[model]}")
        print(f"{model.name} total_times: {total_times[model]}")
        print()

    return metrics, total_times


def plot_metrics(metrics, total_times):
    import matplotlib.pyplot as plt

    ERROR_MSG = "Error: metrics and total_times have different keys"
    models = metrics.keys() # get the models to plot
    assert models == total_times.keys(), ERROR_MSG
    print("Plotting metrics for models {}...".format([model.name for model in models]))
    fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(10, 6))
    for model in models:
        # For now, just make datastructures for test ppls and training times
        test_ppls, training_times = [], []    
        fractions = metrics[model].keys() # extract the list of fractions
        assert fractions == total_times[model].keys(), ERROR_MSG
        for fraction in fractions:
            test_ppls.append(metrics[model][fraction][0])
            training_times.append(total_times[model][fraction][0])
        ax[0].plot(fractions, test_ppls, label=model.name)
        ax[1].plot(fractions, training_times, label=model.name) # this is a wrong plot
    ax[0].set_title("Test Perplexity For Different Data Sizes")
    ax[0].set_xlabel("Fraction of data used")
    ax[0].set_ylabel("Test ppl")
    ax[0].legend()
        
    ax[1].set_title("Training Times For Different Data Sizes")
    ax[1].set_xlabel("Fraction of data used")
    ax[1].set_ylabel("Training time (s)")
    ax[1].legend()

    plt.savefig("test_plot2")
    
    plt.show()
    
    
