class CodeChunker:
    """
    Extracts functions and methods from a Tree-sitter AST.
    """

    def chunk_tree(self, tree, source_code):
        chunks = []

        self._walk_node(
            tree.root_node,
            source_code,
            chunks,
            class_name=None
        )

        return chunks

    def _walk_node(
        self,
        node,
        source_code,
        chunks,
        class_name=None
    ):

        # Handle classes
        if node.type == "class_definition":

            name_node = node.child_by_field_name("name")

            current_class = (
                name_node.text.decode("utf-8")
                if name_node
                else None
            )

            for child in node.children:
                self._walk_node(
                    child,
                    source_code,
                    chunks,
                    current_class
                )

            return

        # Handle functions/methods
        if node.type == "function_definition":

            chunk = self._create_chunk(
                node,
                source_code,
                class_name
            )

            chunks.append(chunk)

            return

        # Continue searching
        for child in node.children:

            self._walk_node(
                child,
                source_code,
                chunks,
                class_name
            )

    def _create_chunk(
        self,
        node,
        source_code,
        class_name
    ):

        name_node = node.child_by_field_name("name")

        function_name = (
            name_node.text.decode("utf-8")
            if name_node
            else "unknown"
        )

        code = source_code[
            node.start_byte:node.end_byte
        ].decode("utf-8")

        return {
            "function_name": function_name,
            "class_name": class_name,
            "code": code,
            "start_line": node.start_point[0] + 1,
            "end_line": node.end_point[0] + 1
        }