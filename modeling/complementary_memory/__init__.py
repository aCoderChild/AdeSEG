"""SAM2 memory study components, imported lazily to avoid unused dependencies."""

__all__ = ["ComplementaryMemoryReader", "NativeComplementaryRead"]


def __getattr__(name):
    if name == "ComplementaryMemoryReader":
        from .reader import ComplementaryMemoryReader
        return ComplementaryMemoryReader
    if name == "NativeComplementaryRead":
        from .integration import NativeComplementaryRead
        return NativeComplementaryRead
    raise AttributeError(name)
