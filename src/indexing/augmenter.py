class CodeAugmenter:
    """
    Creates a searchable textual representation of a code chunk.
    """

    def augment(self, chunk, metadata, file_path, language):
        parts = []

        parts.append(f"FILE: {file_path}")
        parts.append(f"LANGUAGE: {language}")

        if metadata.get("class_name"):
            parts.append(
                f"CLASS: {metadata['class_name']}"
            )

        if metadata.get("function_name"):
            parts.append(
                f"FUNCTION: {metadata['function_name']}"
            )

        if metadata.get("parameters"):
            parts.append(
                "PARAMETERS: "
                + ", ".join(metadata["parameters"])
            )

        if metadata.get("docstring"):
            parts.append(
                f"PURPOSE: {metadata['docstring']}"
            )

        if metadata.get("calls"):
            parts.append(
                "CALLS: "
                + ", ".join(metadata["calls"])
            )

        parts.append("CODE:")
        parts.append(chunk["code"])

        return "\n".join(parts)