import traceback, sys

# Direct file import works -- let's test package import  
print("Testing agents.layer1 package import...")
try:
    from agents.layer1.data_analyst import DataAnalystAgent
    print("data_analyst OK")
except Exception:
    traceback.print_exc()

try:
    from agents.layer1.contextualizer import ContextualizerAgent
    print("contextualizer OK")
except Exception:
    traceback.print_exc()

try:
    from agents.layer1.feature_engineer import FeatureEngineerAgent
    print("feature_engineer OK")
except Exception:
    traceback.print_exc()

print("Testing from agents.layer1 (package __init__)...")
try:
    from agents.layer1 import DataAnalystAgent
    print("DataAnalystAgent OK")
except Exception:
    traceback.print_exc()
