class BridgeConfig:
    def __init__(self, mode, ibkr_host, ibkr_port, client_id, enable_paper_trading):
        self.mode = mode
        self.ibkr_host = ibkr_host
        self.ibkr_port = ibkr_port
        self.client_id = client_id
        self.enable_paper_trading = enable_paper_trading

class BridgeMode:
    IB_INSYNC_ONLY = "ib_insync_only"

class NautilusIBKRBridge:
    def __init__(self, config):
        self.config = config

    async def initialize(self):
        return False

    async def get_account_summary(self):
        return {}

    async def get_positions(self):
        return []

    async def get_market_data(self, symbol):
        return {}

    async def disconnect(self):
        pass

def get_nautilus_ibkr_bridge(config):
    return NautilusIBKRBridge(config)