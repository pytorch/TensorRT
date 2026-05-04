"""Helpers for resolving local TensorRT include and library layouts."""

def one_of(paths):
    """Returns the single existing path, or the first candidate if none exist.

    Args:
      paths: Candidate file paths for the same TensorRT artifact.

    Returns:
      The single matching path, or the first candidate when no path exists.
    """
    matches = native.glob(paths, allow_empty = True)
    if len(matches) > 1:
        fail("Multiple matching paths found: {}".format(matches))
    if len(matches) == 1:
        return matches[0]
    return paths[0]

def any_of(paths, exclude = []):
    """Returns all matching paths across the supported local TensorRT layouts.

    Args:
      paths: Candidate glob patterns spanning supported TensorRT layouts.
      exclude: Optional glob patterns to exclude from the result.

    Returns:
      A list of matching paths.
    """
    return native.glob(paths, allow_empty = True, exclude = exclude)
