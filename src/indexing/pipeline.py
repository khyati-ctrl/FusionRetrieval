import json
from pathlib import Path

from .parser import CodeParser
from .chunker import CodeChunker
from .metadata import MetadataExtractor
from .augmenter import CodeAugmenter
from .hashing import calculate_content_hash


class IndexingPipeline:
    """
    End-to-end code indexing pipeline.

    Flow:
        Source file
            ↓
        Tree-sitter parser
            ↓
        AST-aware chunking
            ↓
        Metadata extraction
            ↓
        Searchable text augmentation
            ↓
        Content hashing
            ↓
        Structured chunks
    """

    def __init__(self):

        self.parser = CodeParser()
        self.chunker = CodeChunker()
        self.metadata_extractor = MetadataExtractor()
        self.augmenter = CodeAugmenter()

    def process_file(self, file_path, language="python"):
        """
        Process a single source file and return structured chunks.
        """

        file_path = Path(file_path)

        # Parse source code
        tree, source_code = self.parser.parse_file(
            file_path
        )

        # Extract functions/classes/methods
        chunks = self.chunker.chunk_tree(
            tree,
            source_code
        )

        results = []

        for index, chunk in enumerate(chunks):

            # Extract metadata
            metadata = self.metadata_extractor.extract(
                chunk
            )

            # Create searchable representation
            augmented_text = self.augmenter.augment(
                chunk,
                metadata,
                str(file_path),
                language
            )

            # Generate content hash
            content_hash = calculate_content_hash(
                chunk["code"]
            )

            # Create final structured chunk
            result = {
                "chunk_id": (
                    f"{file_path.stem}_"
                    f"{metadata['function_name']}_"
                    f"{index}"
                ),
                "file_path": str(file_path),
                "language": language,
                "class_name": metadata["class_name"],
                "function_name": metadata["function_name"],
                "parameters": metadata["parameters"],
                "docstring": metadata["docstring"],
                "calls": metadata["calls"],
                "code": chunk["code"],
                "start_line": chunk["start_line"],
                "end_line": chunk["end_line"],
                "augmented_text": augmented_text,
                "content_hash": content_hash
            }

            results.append(result)

        return results

    def save_chunks(self, chunks, output_path):
        """
        Save chunks as JSON Lines (.jsonl).

        Each line contains one complete code chunk.
        """

        output_path = Path(output_path)

        # Create parent directories if necessary
        output_path.parent.mkdir(
            parents=True,
            exist_ok=True
        )

        with output_path.open(
            "w",
            encoding="utf-8"
        ) as file:

            for chunk in chunks:

                file.write(
                    json.dumps(
                        chunk,
                        ensure_ascii=False
                    )
                    + "\n"
                )


if __name__ == "__main__":

    pipeline = IndexingPipeline()

    # Process test file
    chunks = pipeline.process_file(
        "test.py"
    )

    # Save processed chunks
    output_path = "data/chunks/chunks.jsonl"

    pipeline.save_chunks(
        chunks,
        output_path
    )

    print(
        f"Successfully indexed {len(chunks)} chunks."
    )

    print(
        f"Saved to: {output_path}"
    )