import os
import os.path
import re


def file_ids(directory):
    file_ids = []
    for dirpath, dirnames, filenames in os.walk(directory):
        for filename in [f for f in filenames if f.endswith(".csv")]:
            file_ids.append((os.path.join(dirpath, filename),
                             re.sub(r"^\.\./[\w]*/", "", dirpath),
                             re.sub(r"\.csv", "", filename)))
    return file_ids


def parse_file(file_id, id):
    with open(file_id[0], "rb") as f:
        content = f.read().splitlines(True)[1:]
    content = [re.sub("\n", "", content[i]) + "," + file_id[1] + "," +
               file_id[2] + "," + str(i) + "\n"
               for i in range(len(content))]
    with open("%s.csv" % (id + 1), "wb") as f:
        f.writelines(content)
    return None


if __name__ == "__main__":
    # SAMPLE_FILE = "drivers2/1/1.csv"
    X = file_ids("../drivers2")

    for i in range(len(X)):
        parse_file(X[i], i)
