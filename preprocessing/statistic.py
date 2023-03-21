import json

DATASET_PATH = 'dataset'
MARKS_PATH = './preprocessing/marks.json'
E2E_RESULT_PATH = './E2E_Dataset.json'


def reverse_dict(d):
    return {v: k for k, v in d.items()}


def get_stats():
    stats = {}
    with open(MARKS_PATH) as marks_file:
        marks = json.load(marks_file)
    marks = reverse_dict(marks)

    with open(E2E_RESULT_PATH) as result_file:
        result = json.load(result_file)
    for sentence in result.values():
        for c in sentence:
            if c in marks:
                stats[marks[c]] = stats.get(marks[c], 0) + 0.5

    print(stats)
    return


if __name__ == "__main__":
    get_stats()
