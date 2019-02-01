import json
import shutil
import sys

from allennlp.commands import main

config_file = 'tests/06_b_paper_classifier.json'

overrides = json.dumps({"trainer":{"cuda_device":0}})

serialization_dir = "/tmp/debugger_train"

# whipe out files
shutil.rmtree(serialization_dir,ignore_errors=True)

#Assemble the command into sys.argv

sys.argv = [
    "allennlp",
    "train",
    config_file,
    "-s", serialization_dir,
    "--include-package","my_library",
    "-o", overrides,
]

main()

