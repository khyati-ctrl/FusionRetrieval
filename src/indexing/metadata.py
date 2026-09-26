import re


class MetadataExtractor:
    """
    Extract useful metadata from a code chunk.
    """

    def extract(self, chunk):
        code = chunk["code"]

        metadata = {
            "function_name": chunk.get("function_name"),
            "class_name": chunk.get("class_name"),
            "parameters": self._extract_parameters(code),
            "docstring": self._extract_docstring(code),
            "calls": self._extract_calls(code)
        }

        return metadata

    def _extract_parameters(self, code):
        """
        Extract parameters from the function definition.
        """

        match = re.search(
            r"def\s+\w+\s*\((.*?)\)",
            code,
            re.DOTALL
        )

        if not match:
            return []

        parameters = match.group(1)

        if not parameters.strip():
            return []

        result = []

        for parameter in parameters.split(","):
            parameter = parameter.strip()

            # Remove default values
            parameter = parameter.split("=")[0].strip()

            # Remove self / cls
            if parameter in ("self", "cls"):
                continue

            # Remove type annotation
            parameter = parameter.split(":")[0].strip()

            if parameter:
                result.append(parameter)

        return result

    def _extract_docstring(self, code):
        """
        Extract the first string after the function definition.
        """

        match = re.search(
            r'def\s+\w+\s*\(.*?\):\s*["\']{3}(.*?)["\']{3}',
            code,
            re.DOTALL
        )

        if match:
            return match.group(1).strip()

        return ""

    def _extract_calls(self, code):
        """
        Extract simple function/method calls.
        """

        matches = re.findall(
            r'\b([a-zA-Z_]\w*)\s*\(',
            code
        )

        # Remove the function's own definition from detected calls
        function_name = re.search(
            r'def\s+([a-zA-Z_]\w*)\s*\(',
            code
        )

        if function_name:
            matches = [
                match
                for match in matches
                if match != function_name.group(1)
            ]

        ignored = {
            "if",
            "for",
            "while",
            "return",
            "def",
            "print"
        }

        calls = []

        for match in matches:
            if match not in ignored:
                calls.append(match)

        # Remove duplicates while preserving order
        return list(dict.fromkeys(calls))