import requests
import json

webhook_url_default = 'http://localhost:5000/webhook'  # Replace with your actual webhook URL

def send_webhook_update(action: str, webhook_url: str = webhook_url_default, username: str = 'API Bot'):
    """
    Choose message content via a switch (pattern matching) on `action`, then send to webhook.
    Returns (success: bool, response_or_exception).
    """
    # Python 3.10+ match statement used as a switch
    match action:
        case 'like':
            content = 'agree'
        case 'unlike':
            content = 'reject'
        case 'superlike':
            content = 'superlike'
        case _:
            content = f'Unknown action: {action}'

    data = {'content': content, 'username': username}

    try:
        response = requests.post(webhook_url, json=data, headers={'Content-Type': 'application/json'})
        if response.status_code in (200, 201, 204):
            return True, response
        return False, response
    except requests.RequestException as exc:
        return False, exc

# Optional: make the function available as the module's "default" export
__all__ = ['send_webhook_update']
