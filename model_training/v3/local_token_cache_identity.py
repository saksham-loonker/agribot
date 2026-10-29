"""Fingerprint the frozen token calculation separately from NPY header handling."""
import ast,hashlib


def token_calculation_identity(source):
    tree=ast.parse(source)
    blocks=[node for node in ast.walk(tree) if isinstance(node,ast.With) and any(isinstance(item.context_expr,ast.Call) and isinstance(item.context_expr.func,ast.Attribute) and item.context_expr.func.attr=='inference_mode' for item in node.items)]
    if len(blocks)!=1:raise ValueError('Expected one explicit frozen token calculation')
    return hashlib.sha256(ast.dump(blocks[0],include_attributes=False).encode()).hexdigest()
