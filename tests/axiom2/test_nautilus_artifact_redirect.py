from urllib.request import Request
from scripts.axiom2_restore_nautilus_artifact import ArtifactRedirect


def test_github_token_is_removed_from_storage_redirect():
    handler = ArtifactRedirect()
    request = Request('https://api.github.com/repos/Raynergy-svg/ml_engine/actions/artifacts/11454954090/zip', headers={'Authorization': 'Bearer private'})
    redirected = handler.redirect_request(request, None, 302, 'Found', {}, 'https://approved.blob.core.windows.net/artifact.zip?signature=opaque')
    assert redirected.get_header('Authorization') is None
