from sqlalchemy import Column, DateTime, Float, Integer, String, Text
from sqlalchemy.orm import declarative_base

Base = declarative_base()


class AnomalyEvent(Base):
    """ORM model for a detected anomaly event."""

    __tablename__ = "anomaly_events"

    id = Column(Integer, primary_key=True, autoincrement=True)
    # Event onset (capture time); end/duration added with event segmentation
    timestamp = Column(DateTime(timezone=True), nullable=False)
    end_timestamp = Column(DateTime(timezone=True), nullable=True)
    duration_s = Column(Float, nullable=True)
    anomaly_score = Column(Float, nullable=False)
    # Relative path to the event directory under events_dir
    event_dir = Column(String(512), nullable=False, unique=True)
    # NULL until the embedding is stored and indexed in FAISS
    faiss_index_id = Column(Integer, nullable=True)
    audio_path = Column(String(512), nullable=True)
    frame_path = Column(String(512), nullable=True)
    embedding_path = Column(String(512), nullable=True)
    # Bounding boxes from vision module, serialized as JSON array
    source_region_json = Column(Text, nullable=True)
    # Extra metadata: window_index, feature norms, etc.
    extra_json = Column(Text, nullable=True)

    # Top contributing audio features (z-score explainability), JSON array.
    # Databases written by v0.3 may also hold fusion/video columns
    # (audio_score, video_score, combined_score, fast_*/slow_*,
    # top_video_features, dominant_modality); they are nullable, no longer
    # mapped, and simply ignored (ADR-0015).
    top_audio_features = Column(Text, nullable=True)
