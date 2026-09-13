import numpy as np
import os
import pickle
import datetime
import json
from pathlib import Path


def _output_root(args):
    return Path(getattr(args, "artifact_dir", "artifacts")) / "model_outputs"


def save_loss_to_file(args, arr, name, extension=""):
    filename = get_output_path(args, directory="logging", filename=name, extension=extension, file_type="txt")
    np.savetxt(filename, arr)


def save_predictions_to_file(arr, args, extension=""):
    filename = get_output_path(args, directory="predictions", filename="p", extension=extension, file_type="npy")
    np.save(filename, arr)


def save_model_to_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    pickle.dump(model, open(filename, 'wb'))


def load_model_from_file(model, args, extension=""):
    filename = get_output_path(args, directory="models", filename="m", extension=extension, file_type="pkl")
    return pickle.load(open(filename, 'rb'))


def save_preprocessor_to_file(transformer, args, extension=""):
    """Persist the transformer fitted within one training fold."""
    filename = get_output_path(
        args,
        directory="models",
        filename="preprocessor",
        extension=extension,
        file_type="pkl",
    )
    with open(filename, "wb") as handle:
        pickle.dump(transformer, handle)


def load_preprocessor_from_file(args, extension=""):
    """Load the transformer paired with a fold-specific model."""
    filename = get_output_path(
        args,
        directory="models",
        filename="preprocessor",
        extension=extension,
        file_type="pkl",
    )
    with open(filename, "rb") as handle:
        return pickle.load(handle)


def save_results_to_json_file(args, jsondict, resultsname, append=True):
    """ Write the results to a json file. 
        jsondict: A dictionary with results that will be serialized.
        If append=True, the results will be appended to the original file.
        If not, they will be overwritten if the file already exists. 
    """
    filename = get_output_path(args, filename=resultsname, file_type="json")
    if append:
        if os.path.exists(filename):
            old_res = json.load(open(filename))
            for k, v in jsondict.items():
                old_res[k].append(v)
        else:
            old_res = {}
            for k, v in jsondict.items():
                old_res[k] = [v]
        jsondict = old_res
    json.dump(jsondict, open(filename, "w"))


def save_results_to_file(args, results, train_time=None, test_time=None, best_params=None):
    filename = get_output_path(args, filename="results", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write(args.model_name + " - " + args.dataset + "\n\n")

        for key, value in results.items():
            if isinstance(value, (int, float, np.number)):
                text_file.write("%s: %.5f\n" % (key, value))
            else:
                text_file.write("%s: %s\n" % (key, str(value)))

        if train_time:
            text_file.write("\nTrain time: %f\n" % train_time)

        if test_time:
            text_file.write("Test time: %f\n" % test_time)

        if best_params:
            text_file.write("\nBest Parameters: %s\n\n\n" % best_params)
        
        text_file.write("\n---------------------------------------\n")


def save_hyperparameters_to_file(args, params, results, time=None):
    filename = get_output_path(args, filename="hp_log", file_type="txt")

    with open(filename, "a") as text_file:
        text_file.write(str(datetime.datetime.now()) + "\n")
        text_file.write("Parameters: %s\n\n" % params)
  
        for key, value in results.items():
            if isinstance(value, (int, float, np.number)):
                text_file.write("%s: %.5f\n" % (key, value))
            else:
                text_file.write("%s: %s\n" % (key, str(value)))

        if time:
            text_file.write("\nTrain time: %f\n" % time[0])
            text_file.write("Test time: %f\n" % time[1])

        text_file.write("\n---------------------------------------\n")


def get_output_path(args, filename, file_type, directory=None, extension=None):
    """Build an output path under the configured artifact directory."""
    output_dir = _output_root(args)
    if args.data_type == 'classic':
        run_name = f"{args.dataset}_{args.disease_type}_{args.data_type}"
    else:
        run_name = f"{args.dataset}_{args.disease_type}_{args.data_type}_{args.protein_source}"
    dir_path = output_dir / args.model_name / run_name

    if directory:
        dir_path = dir_path / directory

    dir_path.mkdir(parents=True, exist_ok=True)

    suffix = f"_{extension}" if extension is not None else ""
    return str(dir_path / f"{filename}{suffix}.{file_type}")
