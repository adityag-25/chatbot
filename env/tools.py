def process_refund(order_id):
    return {"status": "success", "message": "Refund processed"}

def escalate_ticket(issue):
    return {"status": "escalated", "message": "Issue escalated"}

def fetch_order(order_id):
    return {"status": "shipped", "message": "Order in transit"}