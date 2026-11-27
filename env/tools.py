"""
Tool simulations for the CustomerSupportEnv.

Each tool returns a dict representing the backend action result.
"""


def process_refund(order_id: str) -> dict:
    """Simulate triggering a refund for a given order."""
    return {
        "tool": "process_refund",
        "order_id": order_id,
        "status": "success",
        "message": f"Refund for order #{order_id} has been initiated. "
                   f"Amount will be credited within 3-5 business days.",
    }


def escalate_ticket(issue: str, order_id: str = "N/A") -> dict:
    """Simulate escalating an issue to the support team."""
    return {
        "tool": "escalate_ticket",
        "order_id": order_id,
        "issue": issue,
        "status": "escalated",
        "message": f"Ticket escalated to Tier-2 support. "
                   f"Our team will reach out within 24 hours regarding: {issue}.",
    }


def order_lookup(order_id: str, order_status: str = "unknown") -> dict:
    """
    Simulate looking up an order from the backend.
    The environment passes the task's actual order_status so the response
    is realistic even in the simulated environment.
    """
    status_messages = {
        "shipped": f"Order #{order_id} has been shipped and is in transit.",
        "delivered": f"Order #{order_id} has been delivered to the address on file.",
        "processing": f"Order #{order_id} is currently being processed.",
        "cancelled": f"Order #{order_id} has been cancelled.",
        "pending": f"Order #{order_id} is pending confirmation.",
        "unknown": f"Order #{order_id} status could not be determined. "
                   f"Please verify the order ID.",
    }
    message = status_messages.get(order_status, status_messages["unknown"])
    return {
        "tool": "order_lookup",
        "order_id": order_id,
        "status": order_status,
        "message": message,
    }
