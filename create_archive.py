import json
import os
import sys
import tarfile
import tempfile
from pathlib import Path

from luke.utils.model_utils import METADATA_FILE, MODEL_FILE, get_entity_vocab_file_path


def create_model_archive(model_file: str, out_file: str, compress: str):
    model_dir = os.path.dirname(model_file)
    json_file = os.path.join(model_dir, METADATA_FILE)
    with open(json_file) as f:
        model_data = json.load(f)
        if "arguments" in model_data:
            del model_data["arguments"]

    file_ext = ".tar" if not compress else ".tar." + compress
    if not out_file.endswith(file_ext):
        out_file = out_file + file_ext

    with tarfile.open(out_file, mode="w:" + compress) as archive_file:
        archive_file.add(model_file, arcname=MODEL_FILE)

        vocab_file_path = get_entity_vocab_file_path(model_dir)
        archive_file.add(vocab_file_path, arcname=Path(vocab_file_path).name)

        with tempfile.NamedTemporaryFile(mode="w") as metadata_file:
            json.dump(model_data, metadata_file, indent=2)
            metadata_file.flush()
            os.fsync(metadata_file.fileno())
            archive_file.add(metadata_file.name, arcname=METADATA_FILE)


model_file = sys.argv[1]
output_file = sys.argv[2]
create_model_archive(model_file, output_file, compress='gz')
