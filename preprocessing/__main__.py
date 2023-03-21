import os
import json

from AzureASR import AzureASR
from AzureNER import AzureNER

DATASET_PATH = 'dataset'
MARKS_PATH = './preprocessing/marks.json'
E2E_RESULT_PATH = './E2E_Dataset.json'


def preprocessing():
    azure_asr = AzureASR()
    azure_ner = AzureNER()

    # Load marks dictionary
    with open(MARKS_PATH) as marks_file:
        marks = json.load(marks_file)

    # Load existing results
    if os.path.isfile(E2E_RESULT_PATH):
        with open(E2E_RESULT_PATH) as result_file:
            result = json.load(result_file)
    else:
        result = {}

    # Determine the number of files to process
    total_files = len(os.listdir(DATASET_PATH))
    num_files_to_process = round(total_files * 0.7) + 1

    # Iterate over dataset files
    for filename in os.listdir(DATASET_PATH):
        if num_files_to_process == 0:
            break

        # Skip files that have already been processed
        if filename in result:
            continue

        full_path = os.path.join(DATASET_PATH, filename)

        # Perform speech-to-text transcription
        transcript = azure_asr.recognize(full_path)

        # Perform named entity recognition
        entities = azure_ner.entity_recognition(transcript)

        # Replace recognized entities with marked versions in the transcript
        for entity in entities:
            mark = marks.get(entity.category, "")
            transcript = transcript.replace(entity.text, f"{mark}{entity.text}{mark}")

        # Store the result for the current file
        result[filename] = transcript

        num_files_to_process -= 1

    # Write the final results to file
    with open(E2E_RESULT_PATH, 'w') as result_file:
        json.dump(result, result_file)

    return


if __name__ == "__main__":
    try:
        preprocessing()
    except Exception as ex:
        print(ex)
