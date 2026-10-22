import os
import sys
import tempfile
from pathlib import Path

# Must be set BEFORE voice_agent.config is imported.
os.environ["VOICE_TOKEN"] = "test-token"
os.environ["VOICE_DATA_DIR"] = tempfile.mkdtemp(prefix="voice-agent-test-")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
