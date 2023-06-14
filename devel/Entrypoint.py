from tiger.cluster import Entrypoint
import config

entrypoint = Entrypoint(config.extdir)
entrypoint.execute()
