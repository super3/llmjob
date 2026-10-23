#!/usr/bin/env python3
# Starts the trader (trader.py). It exists for Railway's builder. Railpack starts a Python service with main.py when
# nothing gives it a start command, and stops the build when it finds none: "No start command detected". The start
# command in railway.toml here reaches it only when the service's config path is set to /trader/railway.toml,
# because Railway's config file doesn't follow the Root Directory. With this file the build works either way.
import sys

import trader

if __name__ == "__main__":
    sys.exit(trader.main())
