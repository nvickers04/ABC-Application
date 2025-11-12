try:
    import langchain.memory
    print("langchain.memory imported successfully")
except ImportError as e:
    print("Import error:", e)