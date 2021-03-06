import numpy as np
import scipy.stats as S
import os
import argparse

from statistics import mean

from get_accuracy import main as get_accuracy



if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder1", type=str, required=True, help="Folder containing the logs")
    parser.add_argument("--log_file_pattern1", type=str, default=".+", help="Pattern for the log files")
    parser.add_argument("--string_accuracy1", type=str, default="Test |  Acc:", help="String to search for accuracy")
    parser.add_argument("--accuracy_marker1", type=str, default="Acc:", help="Field marking accuracy")
    parser.add_argument("--field_separator1", type=str, default=" ", help="Field separator")
    parser.add_argument("--index_from_marker1", type=int, default=None, help="Number of chars from accuracy_marker where the accuracy float value begins. Alternative if --field_speparator is not useful.")
    parser.add_argument("--accuracy_field_length1", type=int, default=None, help="Length of the accuracy field. Used only when --index_from_marker is not None")
    parser.add_argument("--logs_folder2", type=str, required=True, help="Folder containing the logs")
    parser.add_argument("--log_file_pattern2", type=str, default=None, help="Pattern for the log files")
    parser.add_argument("--string_accuracy2", type=str, default=None, help="String to search for accuracy")
    parser.add_argument("--accuracy_marker2", type=str, default=None, help="Field marking accuracy")
    parser.add_argument("--field_separator2", type=str, default=None, help="Field separator")
    parser.add_argument("--index_from_marker2", type=int, default=None, help="Number of chars from accuracy_marker where the accuracy float value begins. Alternative if --field_speparator is not useful.")
    parser.add_argument("--accuracy_field_length2", type=int, default=None, help="Length of the accuracy field. Used only when --index_from_marker is not None")
    args = parser.parse_args()

    if args.log_file_pattern2 is None:
        args.log_file_pattern2 = args.log_file_pattern1
    if args.string_accuracy2 is None:
        args.string_accuracy2 = args.string_accuracy1
    if args.accuracy_marker2 is None:
        args.accuracy_marker2 = args.accuracy_marker1
    if args.field_separator2 is None:
        args.field_separator2 = args.field_separator1
    if args.index_from_marker2 is None:
        args.index_from_marker2 = args.index_from_marker1
    if args.accuracy_field_length2 is None:
        args.accuracy_field_length2 = args.accuracy_field_length1
    
    accuracies1 = np.array(get_accuracy(args.log_file_pattern1, args.logs_folder1, args.string_accuracy1, args.accuracy_marker1, args.field_separator1, args.index_from_marker1, args.accuracy_field_length1))
    accuracies2 = np.array(get_accuracy(args.log_file_pattern2, args.logs_folder2, args.string_accuracy2, args.accuracy_marker2, args.field_separator2, args.index_from_marker2, args.accuracy_field_length2))
    
    print("Accuracy for folder 1", mean(accuracies1))
    print("Accuracy for folder 2", mean(accuracies2))

    ttest = S.ttest_ind(accuracies1, accuracies2, equal_var=False)
    print(f"T-test: {ttest}")

    

