from config import create_client
import gin

gin.parse_config_file("sicim/config.gin")  # or however you load your gin config
client = create_client()

# Test basic connectivity
try:
    models = client.list()
    print("Connected! Available models:", models)
except Exception as e:
    print("Connection failed:", e)