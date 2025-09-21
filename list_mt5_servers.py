import MetaTrader5 as mt5
import pandas as pd

# Initialize MT5
if not mt5.initialize():
    print("MT5 initialization failed")
    mt5.shutdown()
    exit()

# Get list of all servers
servers = mt5.servers_get()

if servers is not None:
    print("\nAvailable MT5 Servers:")
    for i, server in enumerate(servers, 1):
        print(f"{i}. {server.name}")

# Shutdown MT5 connection
mt5.shutdown()
