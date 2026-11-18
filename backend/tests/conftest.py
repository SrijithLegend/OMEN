import os

os.environ["OMEN_DB"] = "sqlite://"  # in-memory database for tests
os.environ["OMEN_ADMIN_KEY"] = "test-admin"
