import json
import shutil
from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile

# from temppathlib import TemporaryDirectory
from wasabi import msg

from json2dnaf import json2dnaf

MAIN_DIR = Path("/mnt/c/Users/camie/OneDrive/LRE paper/data")
IN_ZIP = MAIN_DIR / "EventDNA_corpus_alpino_lets_json.zip"
OUT_ZIP = MAIN_DIR / "EventDNA_dnaf_corpus.zip"


def transform(in_zip: Path, out_path: Path) -> None:

    msg.divider("Transforming data")

    with TemporaryDirectory() as temp_in_dir, TemporaryDirectory() as temp_out_dir:

        temp_in_dir = Path(temp_in_dir)
        temp_out_dir = Path(temp_out_dir)

        msg.info("Unpacking corpus...")
        with ZipFile(in_zip) as corpus_z:
            corpus_z.extractall(temp_in_dir)

        temp_in_dir = temp_in_dir / "EventDNA_corpus_alpino_lets_json"
        alpino_dir = temp_in_dir / "1alpino_xml"
        lets_dir = temp_in_dir / "2lets_tsv"
        webanno_dir = temp_in_dir / "3webanno_json"

        # print(list(alpino_dir.iterdir()))

        msg.info("Processing files...")
        id_generator = (f"eventdna_{n}" for n in range(1, 10000))
        for webanno_json_file in webanno_dir.iterdir():
            current_id = next(id_generator)
            # msg.text("Processing {}".format(webanno_json_file.stem))

            # Grab LETS and Alpino files for this article.
            lets_file = list(lets_dir.glob(f"{webanno_json_file.stem}.*"))
            assert len(lets_file) == 1
            lets_file = lets_file[0]
            alpino_filedir = list(alpino_dir.glob(f"{webanno_json_file.stem}"))
            assert len(alpino_filedir) == 1
            alpino_filedir = alpino_filedir[0]

            # Convert the webanno json to dnaf.
            try:
                dnaf = json2dnaf(webanno_json_file, current_id)
            except Exception as e:
                msg.warn(f"Error processing file: <{webanno_json_file.name}>")
                msg.text(e)

            # Write out the resulting files and zip the archive.
            article_out_dir = temp_out_dir / current_id
            article_out_dir.mkdir()
            with open(article_out_dir / "dnaf.json", "w", newline="") as out:
                json.dump(dnaf, out, indent=4, sort_keys=True)
            lets_file.rename(article_out_dir / "lets.csv")
            alpino_filedir.rename(article_out_dir / "alpino")

        msg.info("Writing archive...")
        # The out_path does not need a suffix. Remove it if one is passed.
        if out_path.suffix == ".zip":
            out_path = out_path.with_suffix("")
        shutil.make_archive(out_path, "zip", temp_out_dir)

        msg.good("All done.")


if __name__ == "__main__":
    transform(IN_ZIP, OUT_ZIP)
