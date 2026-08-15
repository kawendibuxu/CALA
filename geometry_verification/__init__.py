__all__ = ["CrossToken3DVerificationResult", "GeometryVerificationResult", "VGGTGeometryVerifier"]


def __getattr__(name):
    if name in __all__:
        from .vggt_verifier import CrossToken3DVerificationResult, GeometryVerificationResult, VGGTGeometryVerifier

        return {
            "CrossToken3DVerificationResult": CrossToken3DVerificationResult,
            "GeometryVerificationResult": GeometryVerificationResult,
            "VGGTGeometryVerifier": VGGTGeometryVerifier,
        }[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
