from pathlib import Path

from tree_sitter import Language, Parser
import tree_sitter_python as ts_python


class CodeParser:
    """
    Parses source code using Tree-sitter.
    Currently supports Python.
    """

    def __init__(self):
        python_language = Language(ts_python.language())
        self.parser = Parser(python_language)

    def parse_file(self, file_path):
        """
        Read a source file and return its Tree-sitter AST
        along with the original source code.
        """

        file_path = Path(file_path)

        source_code = file_path.read_bytes()

        tree = self.parser.parse(source_code)

        return tree, source_code


if __name__ == "__main__":

    parser = CodeParser()

    tree, source_code = parser.parse_file("test.py")

    print(tree.root_node)