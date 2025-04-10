class DeadBeatKernel(Exception):
    """Raised when the beat kernel is dead. The user should restart the fyp-backer-end kernel or use demucs"""
    pass

class InvalidMashup(Exception):
    """Raised when the mashup is invalid. The user should check the mashup config and try again"""
    pass

class BadMashup(Exception):
    """An internal error that indicates that the mashup is very likely bad."""
    pass
