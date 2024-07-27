import ast
import copy


class LinenoStripper(ast.NodeTransformer):
    """
    Removes line number information from all nodes.

    This is usefull to ensure that templates don't get wrong line numbers attributed.'
    """

    def visit(self, node: ast.AST):
        new: ast.AST = copy.copy(node)

        for attr in ('lineno', 'col_offset', 'end_lineno', 'end_col_offset'):
            if hasattr(new, attr):
                delattr(new, attr)

        return self.generic_visit(new)
