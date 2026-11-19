import os

os.environ.setdefault("OMEN_DB", "sqlite://")  # in-memory database for tests; CI sets OMEN_DB to run them against PostgreSQL
os.environ["OMEN_ADMIN_KEY"] = "test-admin"
os.environ["OMEN_SUBMIT_LIMIT"] = "1000"  # the rate-limit test lowers it itself
