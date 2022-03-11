import os
import argparse
import re
import io

def match_row_file(pattern, file_stream:io.StringIO):
    file_content = file_stream.readlines()
    for line in file_content:
        if re.match(pattern, line):
            return line

def get_accuracy_from_row(row:str, accuracy_marker:str, field_separator:str):
    fields = row.split(field_separator)
    try:
        accuracy_index = fields.index(accuracy_marker)
        accuracy = float(fields[accuracy_index + 1])
        return accuracy
    except ValueError:
        pass

def main(log_file_pattern, logs_folder, string_accuracy, accuracy_marker, field_separator):
    log_files = [os.path.join(logs_folder, f) for f in os.listdir(logs_folder) if re.match(log_file_pattern, f)]

    accuracies = []
    for log_file in log_files:
        with open(log_file, "r") as f:
            row = match_row_file(string_accuracy, f)
            if row is not None:
                accuracy = get_accuracy_from_row(row, accuracy_marker, field_separator)
                accuracies.append(float(accuracy))
    
    return accuracies

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--logs_folder", type=str, default=".", help="Folder containing the logs")
    parser.add_argument("--log_file_pattern", type=str, default="*", help="Pattern for the log files")
    parser.add_argument("--string_accuracy", type=str, default="Test |  Acc:", help="String to search for accuracy")
    parser.add_argument("--accuracy_marker", type=str, default="Acc:", help="Field marking accuracy")
    parser.add_argument("--field_separator", type=str, default=" ", help="Field separator")
    parser.add_argument("--debug", action="store_true", help="Print debug information")
    args = parser.parse_args()

    if args.debug:
        args.logs_folder = "stat_utils"
        args.log_file_pattern = ".+\.out"
        print("Arguments:")
        print(args)

    log_file_pattern = args.log_file_pattern
    logs_folder = os.path.expanduser(os.path.expandvars(args.logs_folder))

    accuracies = main(log_file_pattern, logs_folder, args.string_accuracy, args.accuracy_marker, args.field_separator)
    print(accuracies)

    
    
