"""Release-state helpers for CoMODAL."""


def release_status():
    """Return the release-contract snapshot used by the reproduction harness."""
    from experiment_manifest import release_contract
    return dict(release_contract(),
                interpretation='Implementation checks only')


def require_training_release(allow_unvalidated=False, acknowledge_stylized_scope=False):
    """No-op gate retained for backward compatibility of training entry points."""
    return
