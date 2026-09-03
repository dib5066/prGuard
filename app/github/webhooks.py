import hashlib    #for hashing algorith SHA256
import hmac     #cryptography library
from fastapi import HTTPException, status
from app.core.config import settings

def verify_signature(payload: bytes, signature_header: str | None) -> bool:
    """Verify GitHub webhook payload signature using GITHUB_WEBHOOK_SECRET.

    Args:
        payload: Raw request body bytes.
        signature_header: The X-Hub-Signature-256 header value.

    Returns:
        True if the signature is valid, False otherwise.
    """
    if not settings.GITHUB_WEBHOOK_SECRET:
        # If secret is not configured, deny requests for security
        return False

    if not signature_header:
        return False

    # Signature header format: 'sha256=HEX_DIGEST'
    if not signature_header.startswith("sha256="):
        return False

    received_signature = signature_header.removeprefix("sha256=")
    webhook_secret = settings.GITHUB_WEBHOOK_SECRET.encode("utf-8")
    
    # Calculate expected signature
    computed_signature = hmac.new(webhook_secret, payload, hashlib.sha256).hexdigest()

    # Time-constant comparison to prevent timing attacks
    return hmac.compare_digest(computed_signature, received_signature)


def verify_webhook_request(payload: bytes, signature_header: str | None) -> None:
    """Helper function to assert signature validity.
    
    Raises:
        HTTPException: If signature verification fails.
    """
    if not verify_signature(payload, signature_header):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid webhook signature verification failed.",
        )
