from src.model import build_pipeline

def test_pipeline_creation():
    pipeline = build_pipeline()
    assert pipeline is not None
