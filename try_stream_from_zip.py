from pathlib import Path
from zipfile import ZipFile
from collections import defaultdict

zip_p = Path("data/eventdna_corpus_2020-13-02.zip")

def stream(zip_p):
    with ZipFile(zip_p) as z:

        files = defaultdict(dict)

        for name in z.namelist():
            name = Path(name)

            parent = name.parent
            if parent.stem in {"dnaf", "lets", "alpino"}:
                files[str(name.stem)][str(parent)] = name

        # Validate.
        assert len(files) == 1773
        for path_dict in files.values():
            assert set(path_dict.keys()) == {"dnaf", "lets", "alpino"}

        for id, paths in files.items():
            yield id, paths["dnaf"], paths["lets"], paths["alpino"]

    # print(files)

    # print(z.namelist())

# def stream(zip_p):
#     with ZipFile(zip_p) as z:
#         for f in z.filelist:
#             yield z

if __name__ == "__main__":
    for s in stream(zip_p):
        print(s)
