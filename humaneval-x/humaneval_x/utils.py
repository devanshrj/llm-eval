import gzip
import json
import os
from typing import *


IMPORT_HELPER = {
    "python": [
        "import math",
        "import re",
        "import sys",
        "import copy",
        "import datetime",
        "import itertools",
        "import collections",
        "import heapq",
        "import statistics",
        "import functools",
        "import hashlib",
        "import numpy",
        "import numpy as np",
        "import string",
        "from typing import *",
        "from collections import *",
    ],
    "go"    : [
        "math",
        "strings",
        "fmt",
        "strconv",
        "time",
        "bytes",
        "regexp",
        "sort",
        "math/rand",
        "crypto/md5",
    ],
    "cpp"   : [
        "#include<stdlib.h>",
        "#include<algorithm>",
        "#include<math.h>",
        "#include<stdio.h>",
        "#include<vector>",
        "#include<string>",
        "#include<climits>",
        "#include<cstring>",
        "#include<iostream>",
    ],
}


LANGUAGE_TAG = {
    "c"            : "// language: C",
    "c++"          : "// language: C++",
    "cpp"          : "// language: C++",
    "c#"           : "// language: C#",
    "csharp"       : "// language: C#",
    "css"          : "/* language: CSS */",
    "cuda"         : "// language: Cuda",
    "dart"         : "// language: Dart",
    "lua"          : "// language: Lua",
    "objectivec"  : "// language: Objective-C",
    "objective-c"  : "// language: Objective-C",
    "objective-c++": "// language: Objective-C++",
    "python"       : "# language: Python",
    "perl"         : "# language: Perl",
    "prolog"       : f"% language: Prolog",
    "swift"        : "// language: swift",
    "lisp"         : "; language: Lisp",
    "java"         : "// language: Java",
    "scala"        : "// language: Scala",
    "tex"          : f"% language: TeX",
    "vue"          : "<!--language: Vue-->",
    "markdown"     : "<!--language: Markdown-->",
    "html"         : "<!--language: HTML-->",
    "php"          : "// language: PHP",
    "js"           : "// language: JavaScript",
    "javascript"   : "// language: JavaScript",
    "typescript"   : "// language: TypeScript",
    "go"           : "// language: Go",
    "shell"        : "# language: Shell",
    "rust"         : "// language: Rust",
    "sql"          : "-- language: SQL",
    "kotlin"       : "// language: Kotlin",
    "vb"           : "' language: Visual Basic",
    "ruby"         : "# language: Ruby",
    "pascal"       : "// language: Pascal",
    "r"            : "# language: R",
    "fortran"      : "!language: Fortran",
    "lean"         : "-- language: Lean",
    "matlab"       : f"% language: Matlab",
    "delphi"       : "{language: Delphi}",
    "scheme"       : "; language: Scheme",
    "basic"        : "' language: Basic",
    "assembly"     : "; language: Assembly",
    "groovy"       : "// language: Groovy",
    "abap"         : "* language: Abap",
    "gdscript"     : "# language: GDScript",
    "haskell"      : "-- language: Haskell",
    "julia"        : "# language: Julia",
    "elixir"       : "# language: Elixir",
    "excel"        : "' language: Excel",
    "clojure"      : "; language: Clojure",
    "actionscript" : "// language: ActionScript",
    "solidity"     : "// language: Solidity",
    "powershell"   : "# language: PowerShell",
    "erlang"       : f"% language: Erlang",
    "cobol"        : "// language: Cobol",
}


def stream_jsonl(filename: str) -> Iterable[Dict]:
    """
    Parses each jsonl line and yields it as a dictionary
    """
    if filename.endswith(".gz"):
        with open(filename, "rb") as gzfp:
            with gzip.open(gzfp, "rt") as fp:
                for line in fp:
                    if any(not x.isspace() for x in line):
                        yield json.loads(line)
    else:
        with open(filename, "r") as fp:
            for line in fp:
                if any(not x.isspace() for x in line):
                    yield json.loads(line)


def read_dataset(
    data_file: str = None,
    dataset_type: str = "humaneval",
    num_shot=None,
) -> Dict:
    if num_shot is not None:
        print(f"{num_shot}-shot setting...")
    if "humaneval" in dataset_type.lower():
        if data_file is None:
            current_path = os.path.dirname(os.path.abspath(__file__))
            data_file = os.path.join(current_path, "..", "python", "data", "humaneval_python.jsonl.gz")
        dataset = {task["task_id"]: task for task in stream_jsonl(data_file)}
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset


def read_translation_dataset(
    data_file_src: str = None,
    data_file_tgt: str = None,
    lang_src: str = None,
    lang_tgt: str = None,
    dataset_type: str = "humaneval",
) -> Dict:
    if "humaneval" in dataset_type.lower():
        dataset_src = {task["task_id"]: task for task in stream_jsonl(data_file_src)}
        dataset_tgt = {task["task_id"].split("/")[-1]: task for task in stream_jsonl(data_file_tgt)}
        for k, sample in dataset_src.items():
            prompt = "code translation\n"
            if lang_src == "cpp":
                prompt += "C++:\n"
            elif lang_src == "js":
                prompt += "JavaScript:\n"
            else:
                prompt += f"{lang_src}:\n".capitalize()
            prompt += dataset_src[k]["declaration"] + "\n" + dataset_src[k]["canonical_solution"].rstrip() + "\n"
            if lang_tgt == "cpp":
                prompt += "C++:\n"
            elif lang_tgt == "js":
                prompt += "JavaScript:\n"
            else:
                prompt += f"{lang_tgt}:\n".capitalize()
            prompt += dataset_tgt[k.split("/")[-1]]["declaration"]
            dataset_src[k]["prompt"] = prompt
    else:
        raise f"Dataset: {dataset_type} not supported."

    return dataset_src


def process_extra_prompt(prompt: str, language_type: str = None) -> str:
    """
    Processes the extra prompt.
    """
    language = language_type.lower()
    if language in LANGUAGE_TAG:
        extra_prompt = LANGUAGE_TAG[language] + "\n"
    else:
        extra_prompt = ""

    return extra_prompt + prompt


def is_code_generation_finished(
    code: str,
    language_type: str = None,
    dataset: str = None,
):
    """
    Checks whether the generated code is finished.
    """
    if language_type is None or dataset is None:
        return False

    if "humaneval" in dataset.lower():
        if language_type.lower() == "python":
            for line in code.split("\n"):
                if len(line.strip()) > 0 and line[0] != ' ' and line[0] != '\t':
                    return True
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint"]
            for w in end_words:
                if w in code:
                    return True
        elif language_type.lower() == "java":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "go":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "js":
            if code.count("{") + 1 == code.count("}"):
                return True
        elif language_type.lower() == "cpp":
            if code.count("{") + 1 == code.count("}"):
                return True

    return False


def cleanup_code(
    code: str,
    language_type: str = None,
    dataset: str = None,
):
    """
    Cleans up the generated code.
    """
    if language_type is None or dataset is None:
        return code

    if "humaneval" in dataset.lower():
        if language_type.lower() == "python":
            end_words = ["\ndef", "\nclass", "\nif", "\n#", "\nprint", "\nassert"]
            for w in end_words:
                if w in code:
                    code = code[:code.rfind(w)]
        elif language_type.lower() == "java":
            main_pos = code.find("public static void main")
            if main_pos != -1:
                code = code[:main_pos] + '}'
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
            if code.count('{') + 1 == code.count('}'):
                code += "\n}"
        elif language_type.lower() == "go":
            end_words = ["\n//", "\nfunc main("]
            for w in end_words:
                if w in code:
                    code = code[:code.rfind(w)]
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "cpp":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'
        elif language_type.lower() == "js":
            if '}' in code:
                code = code[:code.rfind('}')] + '}'

    return code